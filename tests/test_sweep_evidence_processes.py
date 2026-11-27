"""Sibling observed-confidence evidence in worker processes: byte identity, errors, teardown."""
from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from XTA import outputs, sweep_evidence_processes, sweep_views
from XTA import tta_sweep_view_publication as publication_module
from XTA.geometry import ViewInfo
from XTA.sweep_confidence import SweepObservedConfidenceWriter
from XTA.sweep_evidence_processes import (PROCESSES_ENV, ProcessEvidence, SweepEvidenceJob,
                                          SweepEvidenceProcessPool, resolve_native_journal_view,
                                          run_sweep_evidence_job, sweep_evidence_process_count)
from XTA.sweep_views import SweepViewIntegrityError
from XTA.tta_sweep_view_publication import ViewMajorPublication, branch_run_id


CUTS = (50, 70, 100)
SHAPE = (3, 8, 8)
REPO = Path(__file__).resolve().parents[1]


def _view(aug="a0"):
    return ViewInfo(name=f"transverse__tta_{aug}", num_slices=3, src_h=8, src_w=8, pad_mode="pad",
                    family="orthogonal", physical_view_name="transverse", full_t=3, full_h=8,
                    full_w=8, tta_aug_id=aug)


@pytest.fixture(autouse=True)
def producer_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("YOLO_TTA_NRRD_GPU_MIRROR_TEE", "0")
    monkeypatch.setenv("YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN", "0")
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "journal-work"))
    monkeypatch.setenv("XTA_YOLO_SWEEP_ID", "family-one")
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_ROLE", "produce")
    monkeypatch.setenv("XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS", ",".join(map(str, CUTS)))
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_GIB", "1")
    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_VERIFY", "0")
    from XTA import nrrd_compression
    monkeypatch.setattr(nrrd_compression, "_require_nrrd_member_codec",
                        lambda **_: ("zlib", 1, gzip.compress))


def _seal(view, identity="identity-one"):
    """Stage one producer view: scalar-owned floor, two crops per 0.70 frame, empty 1.00."""
    writer = sweep_views.NativeTaskFanoutWriter(None, "detector", view.name, SHAPE, CUTS,
        frame_start=0, expected_frames=SHAPE[0], backend="gpu", identity=identity,
        scalar_owned_cutoff=50)
    rng = np.random.default_rng(len(view.name))
    for z in range(SHAPE[0]):
        mask = np.zeros((4, 6), np.uint8)
        mask[1:3, 1:5] = 1
        mask[0, 0] = z % 2
        scores = (rng.integers(1, 256, mask.shape) * mask).astype(np.uint8)
        writer.put(70, z, mask, scores, bbox=(1, 5, 1, 7), prediction_count=2,
                   frames_with_predictions=1)
        writer.put_empty_range(100, z, z + 1, score_lane=True, prediction_counts=[0],
                               frames_with_predictions=[0])
    writer.put_scalar_owned_range(50, 0, SHAPE[0], prediction_counts=np.full(SHAPE[0], 3),
                                  frames_with_predictions=np.ones(SHAPE[0], np.int64))
    decision = sweep_views.seal_view_family("detector", view.name, SHAPE, CUTS,
                                            receipts=[writer.finish()], identity=identity)
    assert decision["status"] == "ready"
    return decision


def _owner(root):
    family, work = root / "family", root / "work"
    branch = family / "confidence_cutoffs" / "conf_050"
    sink = outputs.NrrdLayerSink(nrrd_dir=branch / "nrrd", stem="case", output_shape_tyx=SHAPE,
        max_workers=2, low_quality_specs=[outputs.LowQualityDownbinSpec("0.5", "half", .5, (2, 4, 4))],
        low_quality_root=branch / "low_quality", run_id=branch_run_id("family-one", 50))
    owner = ViewMajorPublication(family_root=family, workspace=work, sweep_id="family-one",
        cutoffs_hundredths=CUTS, current_cutoff=50, scalar_sink=sink, source_shape_tyx=SHAPE,
        model_name="detector", task="segment", min_conf=0., confidence_floor_by_backend={"gpu": .5},
        retain_confidence=True)
    return owner, sink


def _publish(owner, view, decision, identity="identity-one", *, declare_journal=True):
    """Publish like SweepViewSchedule: crops from journal readers, declared by identity."""
    readers = {c: sweep_views.open_view("detector", view.name, c, native_shape=SHAPE, identity=identity)
               for c in CUTS[1:]}
    declared = dict(native_journal_identity=identity) if declare_journal else {}
    try:
        return owner.publish_view_family(view, native_shape=SHAPE,
            read_mask_plane=lambda c, z: readers[c].read_mask_plane(z),
            read_score_plane=lambda c, z: readers[c].read_score_plane(z),
            read_mask_crops=lambda c, z: readers[c].iter_mask_crops(z),
            read_native_crops=lambda c, z: readers[c].iter_crops(z),
            counts_by_cutoff={int(c): int(n) for c, n in decision["prediction_counts"].items()},
            **declared)
    finally:
        for reader in readers.values():
            reader.close()


def _receipts(root):
    return sorted((root / "work/view_publication/published/by_view").glob("*.zip"))


def _wait_for_workers(pool, count):
    deadline = time.monotonic() + 60
    while len(pool.worker_pids) < count and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(pool.worker_pids) == count, pool.worker_pids


def _thread_trees(root, views, decisions, monkeypatch):
    """Reference evidence written on producer threads only."""
    monkeypatch.setenv(PROCESSES_ENV, "0")
    owner, floor = _owner(root)
    try:
        for view, decision in zip(views, decisions):
            _publish(owner, view, decision)
    finally:
        owner.close()
        floor.shutdown()
    return {c: _digests(_evidence_tree(root, c)) for c in CUTS[1:]}


def _digests(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _evidence_tree(root, cutoff):
    return root / "family" / "confidence_cutoffs" / f"conf_{cutoff:03d}" / "reconciliation_evidence"


def test_process_count_default_explicit_and_kill_switch(monkeypatch):
    monkeypatch.delenv(PROCESSES_ENV, raising=False)
    assert sweep_evidence_process_count(16, 4, cpus=128) == 16
    assert sweep_evidence_process_count(2, 4, cpus=16) == 4
    assert sweep_evidence_process_count(2, 1, cpus=4) == 2
    monkeypatch.setenv(PROCESSES_ENV, "5")
    assert sweep_evidence_process_count(16, 4, cpus=128) == 5
    monkeypatch.setenv(PROCESSES_ENV, "0")
    assert sweep_evidence_process_count(16, 4, cpus=128) == 0
    for bad in ("-1", "two"):
        monkeypatch.setenv(PROCESSES_ENV, bad)
        with pytest.raises(ValueError):
            sweep_evidence_process_count(2, 2, cpus=16)


def test_reserved_layer_is_pending_until_completed_or_abandoned(tmp_path):
    writer = SweepObservedConfidenceWriter(tmp_path / "conf_070", cutoff_hundredths=70,
        source_shape_tyx=SHAPE, run_id="run", task="segment", min_conf=0.0,
        confidence_floor_by_backend={"gpu": .7})
    request = writer.begin_view(_view(), "detector", native_shape=SHAPE)
    with pytest.raises(RuntimeError, match="Duplicate"):
        writer.begin_view(_view(), "detector", native_shape=SHAPE)
    with pytest.raises(RuntimeError, match="unfinished"):
        writer.finish()
    writer.abandon_view(request)
    writer.finish()
    again = writer.begin_view(_view(), "detector", native_shape=SHAPE)
    other = writer.begin_view(_view("a1"), "detector", native_shape=SHAPE)
    from XTA.sweep_confidence import write_sweep_view_evidence

    def crops(z):
        yield 2, 4, 2, 5, np.ones((2, 3), np.uint8), np.full((2, 3), 9 + z, np.uint8)

    reference = write_sweep_view_evidence(other, crops)
    with pytest.raises(RuntimeError, match="differs from its reservation"):
        writer.complete_view(again, reference)  # another layer's files
    with pytest.raises(RuntimeError, match="not reserved"):
        writer.complete_view(again, reference)  # the failed adoption released it
    writer.complete_view(other, reference)
    catalogue = json.loads(writer.finish().read_text())
    assert [row["directory"] for row in catalogue["layers"]] == [other.identity]


def test_worker_evidence_is_byte_identical_to_producer_threads(tmp_path, monkeypatch):
    views = [_view("a0"), _view("a1")]
    decisions = [_seal(view) for view in views]
    trees = {}
    for mode, processes in (("threads", "0"), ("processes", "2")):
        monkeypatch.setenv(PROCESSES_ENV, processes)
        root = tmp_path / mode
        owner, floor = _owner(root)
        try:
            assert (owner._evidence_processes is None) == (processes == "0")
            for view, decision in zip(views, decisions):
                refs = _publish(owner, view, decision)
                assert refs[70] is not None and refs[100] is None
            receipts = sorted((root / "work/view_publication/published/by_view").glob("*.zip"))
            assert len(receipts) == 2
            # Worker mode never starts a producer evidence thread.
            assert bool(owner._confidence_pool._threads) == (processes == "0")
        finally:
            owner.close()
            floor.shutdown()
        trees[mode] = {c: _digests(_evidence_tree(root, c)) for c in CUTS[1:]}
    assert trees["threads"] == trees["processes"]
    assert all("manifest.json" in files for files in trees["threads"].values())
    assert len(trees["threads"][70]) == 2 * 3 + 1  # two layers + catalogue


def test_worker_error_propagates_without_poisoning_the_nrrd_barrier(tmp_path, monkeypatch):
    from XTA.cuda_d1 import _nrrd_layer_key
    monkeypatch.setenv(PROCESSES_ENV, "1")
    bad, good = _view("a0"), _view("a1")
    decisions = {view.name: _seal(view) for view in (bad, good)}
    owner, floor = _owner(tmp_path)
    try:
        key = _nrrd_layer_key(view_name=bad.name, source="fullframe", mask_kind="yolo",
                              pass_index=0, stage="pre_interpolation")
        identity = hashlib.sha256(("detector\0" + key).encode("utf-8")).hexdigest()
        blocked = _evidence_tree(tmp_path, 70) / identity
        blocked.mkdir(parents=True)
        (blocked / "metadata.json").write_text("{}")
        with pytest.raises(FileExistsError, match="Confidence evidence already exists") as caught:
            _publish(owner, bad, decisions[bad.name])
        # The worker's traceback travels with the original exception type.
        assert "write_blocks" in str(caught.value.__cause__)
        assert not owner._confidence[70]._pending and not owner._confidence[100]._pending
        assert not list((tmp_path / "work/view_publication/published/by_view").glob("*.zip"))
        # The shared NRRD owner and the worker pool keep serving other views.
        refs = _publish(owner, good, decisions[good.name])
        assert refs[70] is not None
        assert len(list((tmp_path / "work/view_publication/published/by_view").glob("*.zip"))) == 1
    finally:
        owner.close()
        floor.shutdown()


def test_cancel_refuses_new_jobs_and_close_joins_workers(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setenv(PROCESSES_ENV, "2")
    view = _view()
    decision = _seal(view)
    owner, floor = _owner(tmp_path)
    try:
        pool = owner._evidence_processes
        deadline = time.monotonic() + 60
        while len(pool.worker_pids) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        workers = [psutil.Process(pid) for pid in pool.worker_pids]
        assert len(workers) == 2 and all(w.is_running() for w in workers)
        owner.cancel("test failure")
        with pytest.raises(RuntimeError, match="worker processes are closed"):
            _publish(owner, view, decision)
        assert not owner._confidence[70]._pending
    finally:
        owner.close()
        floor.shutdown()
    gone, alive = psutil.wait_procs(workers, timeout=30)
    assert not alive


_ORPHAN_SCRIPT = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from XTA.sweep_evidence_processes import SweepEvidenceProcessPool

if __name__ == "__main__":
    pool = SweepEvidenceProcessPool(2)
    deadline = time.monotonic() + 60
    while len(pool.worker_pids) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    print("WORKERS", *pool.worker_pids, flush=True)
    time.sleep(600)
"""


def test_workers_exit_when_the_producer_is_killed(tmp_path):
    psutil = pytest.importorskip("psutil")
    script = tmp_path / "producer.py"
    script.write_text(_ORPHAN_SCRIPT)
    producer = subprocess.Popen([sys.executable, "-B", str(script), str(REPO)],
                                stdout=subprocess.PIPE, text=True,
                                env={**os.environ, "PYTHONPATH": str(REPO)})
    try:
        line = ""
        while not line.startswith("WORKERS"):  # workers may log their codec first
            line = producer.stdout.readline()
            assert line, "producer exited before reporting its workers"
        pids = [int(value) for value in line.split()[1:]]
        assert len(pids) == 2, line
        workers = [psutil.Process(pid) for pid in pids]
    finally:
        producer.kill()  # no teardown, no atexit: TerminateProcess / SIGKILL
        producer.wait(30)
    gone, alive = psutil.wait_procs(workers, timeout=30)
    assert not alive, [w.pid for w in alive]


def test_undeclared_crops_are_read_on_producer_threads(tmp_path, monkeypatch):
    """Without native_journal_identity the given crops are the evidence source."""
    monkeypatch.setenv(PROCESSES_ENV, "2")
    view = _view()
    decision = _seal(view)
    owner, floor = _owner(tmp_path)
    jobs = []
    try:
        pool = owner._evidence_processes
        real_submit = pool.submit
        pool.submit = lambda *args: jobs.append(args) or real_submit(*args)
        refs = _publish(owner, view, decision, declare_journal=False)
        assert refs[70] is not None
        assert len(_receipts(tmp_path)) == 1
        assert owner._confidence_pool._threads and not jobs
    finally:
        owner.close()
        floor.shutdown()


def test_idle_worker_death_degrades_to_producer_threads(tmp_path, monkeypatch, capsys):
    psutil = pytest.importorskip("psutil")
    views = [_view("a0"), _view("a1")]
    decisions = [_seal(view) for view in views]
    expected = _thread_trees(tmp_path / "threads", views, decisions, monkeypatch)
    fallbacks = []
    monkeypatch.setattr(publication_module, "_record_evidence_fallback", lambda: fallbacks.append(1))
    monkeypatch.setenv(PROCESSES_ENV, "2")
    root = tmp_path / "degraded"
    owner, floor = _owner(root)
    try:
        pool = owner._evidence_processes
        _wait_for_workers(pool, 2)
        workers = [psutil.Process(pid) for pid in pool.worker_pids]
        workers[0].kill()  # an idle worker dies: the executor is broken for good
        deadline = time.monotonic() + 30
        while not pool._executor._broken and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pool._executor._broken
        assert _publish(owner, views[0], decisions[0])[70] is not None
        assert pool.broken and not pool.closed and len(fallbacks) == 2
        # Every worker was joined before a producer thread re-wrote a layer.
        gone, alive = psutil.wait_procs(workers, timeout=0)
        assert not alive
        # Later views go straight to producer threads.
        assert _publish(owner, views[1], decisions[1])[70] is not None
        assert len(fallbacks) == 2
        assert all(not writer._pending for writer in owner._confidence.values())
        assert len(_receipts(root)) == 2
    finally:
        owner.close()
        floor.shutdown()
    assert {c: _digests(_evidence_tree(root, c)) for c in CUTS[1:]} == expected
    assert capsys.readouterr().err.count("Sweep evidence worker processes failed") == 1


class _DyingPool:
    """Stands in for SweepEvidenceProcessPool: every job's worker dies mid-write."""

    instances = []

    def __init__(self, processes):
        self.events, self.closed, self.error = [], False, None
        self.instances.append(self)

    @property
    def broken(self):
        return self.error is not None

    def submit(self, journal, cutoff, request):
        assert journal.identity == "identity-one" and journal.family_id == "family-one"
        # A torn layer of the dead worker: the producer thread must overwrite it.
        request.path.mkdir(parents=True, exist_ok=True)
        (request.path / "scores.u8.zlib.partial").write_bytes(b"torn")
        (request.path / "index.bin.partial").write_bytes(b"torn")
        future = Future()
        future.set_exception(BrokenProcessPool("A child process terminated abruptly"))
        self.events.append(("submitted", cutoff))
        return future

    def mark_broken(self, error):
        self.error = error
        self.events.append(("joined", None))

    def cancel(self):
        self.closed = True

    def close(self):
        self.cancel()


def test_worker_death_mid_view_rewrites_layers_on_threads_after_join(tmp_path, monkeypatch):
    views = [_view("a0"), _view("a1")]
    decisions = [_seal(view) for view in views]
    expected = _thread_trees(tmp_path / "threads", views, decisions, monkeypatch)
    monkeypatch.setenv(PROCESSES_ENV, "2")
    monkeypatch.setattr(sweep_evidence_processes, "SweepEvidenceProcessPool", _DyingPool)
    _DyingPool.instances.clear()
    real_publish = SweepObservedConfidenceWriter.publish_view

    def recorded(writer, *args, **kwargs):
        _DyingPool.instances[0].events.append(("thread-write", writer.cutoff))
        assert threading.current_thread().name.startswith("sweep-confidence")
        return real_publish(writer, *args, **kwargs)

    monkeypatch.setattr(SweepObservedConfidenceWriter, "publish_view", recorded)
    root = tmp_path / "dying"
    owner, floor = _owner(root)
    try:
        pool = owner._evidence_processes
        assert isinstance(pool, _DyingPool)
        for view, decision in zip(views, decisions):
            assert _publish(owner, view, decision)[70] is not None
        events = pool.events
        # Both cutoffs of the first view went to workers; each retry waited for the join.
        assert [e for e in events if e[0] == "submitted"] == [("submitted", 70), ("submitted", 100)]
        first_write = events.index(next(e for e in events if e[0] == "thread-write"))
        assert ("joined", None) in events[:first_write]
        assert sorted(e[1] for e in events if e[0] == "thread-write") == [70, 70, 100, 100]
        assert all(not writer._pending for writer in owner._confidence.values())
        assert len(_receipts(root)) == 2
    finally:
        owner.close()
        floor.shutdown()
    trees = {c: _digests(_evidence_tree(root, c)) for c in CUTS[1:]}
    assert trees == expected  # torn .partial files were replaced, not left behind


def test_broken_job_after_cancel_keeps_its_error_and_starts_no_retry(tmp_path):
    writer = SweepObservedConfidenceWriter(tmp_path / "conf_070", cutoff_hundredths=70,
        source_shape_tyx=SHAPE, run_id="run", task="segment", min_conf=0.0,
        confidence_floor_by_backend={"gpu": .7})
    request = writer.begin_view(_view(), "detector", native_shape=SHAPE)
    pool = _DyingPool(1)
    pool.closed = True
    future = Future()
    future.set_exception(BrokenProcessPool("worker died during teardown"))
    retries = []
    task = ProcessEvidence(writer, request, future, pool=pool, submitted_wall=time.time(),
                           retry=lambda: retries.append(1))
    task.settle()
    assert task.retry is None and not retries and pool.broken
    with pytest.raises(BrokenProcessPool, match="teardown"):
        task.result()
    assert not writer._pending


def test_journal_resolution_uses_the_readers_decision_check(tmp_path, monkeypatch):
    view = _view()
    _seal(view)
    journal = resolve_native_journal_view("detector", view.name, SHAPE, (70, 100), "identity-one")
    assert (journal.identity, journal.family_id, journal.native_shape) == (
        "identity-one", "family-one", SHAPE)
    rejected = [
        ("detector", view.name, SHAPE, (50, 70), "identity-one"),   # scalar-owned sibling
        ("detector", view.name, SHAPE, (70, 100), "identity-two"),  # another identity
        ("detector", view.name, (3, 8, 9), (70, 100), "identity-one"),  # another shape
        ("other", view.name, SHAPE, (70, 100), "identity-one"),     # another model: no decision
        ("detector", view.name, SHAPE, (70, 90), "identity-one"),   # an unsealed cutoff
    ]
    for args in rejected:
        with pytest.raises(SweepViewIntegrityError):
            resolve_native_journal_view(*args)
    monkeypatch.setenv("XTA_YOLO_SWEEP_ID", "family-two")
    with pytest.raises(SweepViewIntegrityError):
        resolve_native_journal_view("detector", view.name, SHAPE, (70, 100), "identity-one")


def test_decision_change_before_the_worker_reads_is_an_integrity_error(tmp_path):
    view = _view()
    _seal(view)
    journal = resolve_native_journal_view("detector", view.name, SHAPE, (70, 100), "identity-one")
    (path,) = (tmp_path / "journal-work" / "native_fanout" / "views").glob("*/metadata.json")
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision["rewritten_after_resolution"] = True  # still a valid decision for every reader
    path.write_text(json.dumps(decision), encoding="utf-8")
    writer = SweepObservedConfidenceWriter(tmp_path / "conf_070", cutoff_hundredths=70,
        source_shape_tyx=SHAPE, run_id="run", task="segment", min_conf=0.0,
        confidence_floor_by_backend={"gpu": .7})
    request = writer.begin_view(view, "detector", native_shape=SHAPE)
    with pytest.raises(SweepViewIntegrityError, match="changed before its evidence worker"):
        run_sweep_evidence_job(SweepEvidenceJob(journal, 70, request))
    assert not (request.path / "metadata.json").exists()


def test_evidence_uses_the_live_family_decision_but_worker_reopens_it(tmp_path, monkeypatch):
    from XTA import sweep_views
    view = _view()
    _seal(view)
    with sweep_views.open_sibling_views("detector", view.name, (50, 70, 100),
                                        native_shape=SHAPE, identity="identity-one") as family:
        with monkeypatch.context() as admission:
            def unexpected(*args, **kwargs):
                raise AssertionError("The parent read its open family's decision twice")
            admission.setattr(sweep_views, "_decision", unexpected)
            journal = resolve_native_journal_view("detector", view.name, SHAPE, (70, 100), "identity-one",
                                                  crops=family.native_crops)
            with pytest.raises(SweepViewIntegrityError):
                resolve_native_journal_view("detector", view.name, SHAPE, (70,), "different",
                                            crops=family.native_crops)
        (path,) = (tmp_path / "journal-work" / "native_fanout" / "views").glob("*/metadata.json")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["changed_after_family_open"] = True
        path.write_text(json.dumps(metadata), encoding="utf-8")
        writer = SweepObservedConfidenceWriter(tmp_path / "conf_070", cutoff_hundredths=70,
            source_shape_tyx=SHAPE, run_id="run", task="segment", min_conf=0.0,
            confidence_floor_by_backend={"gpu": .7})
        request = writer.begin_view(view, "detector", native_shape=SHAPE)
        with pytest.raises(SweepViewIntegrityError, match="changed before its evidence worker"):
            run_sweep_evidence_job(SweepEvidenceJob(journal, 70, request))
        assert not (request.path / "metadata.json").exists()
    with pytest.raises(SweepViewIntegrityError):
        resolve_native_journal_view("detector", view.name, SHAPE, (70,), "identity-one", crops=family.native_crops)


def test_receipts_follow_durable_worker_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv(PROCESSES_ENV, "2")
    view = _view()
    decision = _seal(view)
    real_write = publication_module._write_view_receipts
    checked = []

    def receipts_after_evidence(path, receipts):
        for cent in receipts:
            tree = _evidence_tree(tmp_path, cent)
            (row,) = json.loads((tree / "manifest.json").read_text(encoding="utf-8"))["layers"]
            assert (tree / row["directory"] / "metadata.json").is_file()
        checked.append(list(receipts))
        real_write(path, receipts)

    monkeypatch.setattr(publication_module, "_write_view_receipts", receipts_after_evidence)
    owner, floor = _owner(tmp_path)
    jobs = []
    try:
        pool = owner._evidence_processes
        real_submit = pool.submit
        pool.submit = lambda *args: jobs.append(args[1]) or real_submit(*args)
        _publish(owner, view, decision)
    finally:
        owner.close()
        floor.shutdown()
    assert sorted(jobs) == [70, 100] and checked == [[70, 100]]


def test_affinity_widening_reaches_live_workers_and_their_new_threads(monkeypatch):
    from types import SimpleNamespace
    calls = []
    tasks = {101: [[101, 102], [101, 102, 103]], 202: [[202]]}

    def listdir(path):
        pid = int(path.split('/')[2])
        if pid not in tasks:
            raise FileNotFoundError(path)
        listing = tasks[pid][0]
        if len(tasks[pid]) > 1:
            tasks[pid].pop(0)  # a thread appears during the first pass
        return [str(tid) for tid in listing]

    monkeypatch.setattr(sweep_evidence_processes.os, 'listdir', listdir)
    monkeypatch.setattr(sweep_evidence_processes.os, 'sched_setaffinity',
                        lambda tid, mask: calls.append((tid, tuple(mask))), raising=False)
    live = SimpleNamespace(closed=False, broken=False, worker_pids=[101],
                           _executor=SimpleNamespace(_processes={101: None, 202: None, 303: None}))
    closed = SimpleNamespace(closed=True, broken=False, worker_pids=[404], _executor=None)
    broken = SimpleNamespace(closed=False, broken=True, worker_pids=[505], _executor=None)
    monkeypatch.setattr(sweep_evidence_processes, '_LIVE_POOLS', [live, closed, broken])
    assert sweep_evidence_processes.widen_evidence_worker_affinity([3, 1, 2]) == 4
    # 303 exited (no /proc entry): no bare-pid call that could reach a reused pid.
    assert sorted(calls) == [(101, (1, 2, 3)), (102, (1, 2, 3)), (103, (1, 2, 3)), (202, (1, 2, 3))]


def test_executor_callbacks_never_wait_for_the_submission_lock():
    """A broken executor runs future callbacks under the lock its submit() waits for."""
    pool = SweepEvidenceProcessPool(1)
    try:
        done = Future()
        done.set_result(os.getpid())
        with pool._lock:  # a publisher blocked inside executor.submit
            callbacks = threading.Thread(
                target=lambda: (pool._forget(done), pool._ready(done, 0.0)))
            callbacks.start()
            callbacks.join(10)
            assert not callbacks.is_alive()
    finally:
        pool.close()
