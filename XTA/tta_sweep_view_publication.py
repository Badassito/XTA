"""Publish one native detector view across cutoff directories before retiring its packet.

The producer owns only sibling publications. Its ordinary scalar pipeline still
publishes the floor's masks and confidence. Later scalar children restore their
own published layer refs and execute the normal final-union and output tails.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from collections import Counter
from dataclasses import asdict, fields
from functools import partial
import hashlib
import json
import math
from operator import index
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import NAMESPACE_URL, uuid5

from .json_publication import write_json_atomic


def _cent(value):
    if isinstance(value, bool):
        raise ValueError("A branch cutoff must be integer hundredths")
    cent = index(value)
    if not 0 <= cent <= 100:
        raise ValueError("A branch cutoff must be in [0, 100]")
    return cent


def branch_run_id(sweep_id, cutoff):
    """Use the same unique scalar identity when producing and consuming a view."""
    if not isinstance(sweep_id, str) or not sweep_id.strip():
        raise ValueError("A view publication requires a sweep ID")
    return uuid5(NAMESPACE_URL, f"xta.view-major.branch/1/{sweep_id}/{_cent(cutoff)}").hex


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _file_identity(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


_CVOL_FILES = ("chunks.bin", "index.bin", "meta.json")


def _store_identity(path):
    """Fence all three regular files of a closed CVOL without hashing its voxel data."""
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Expected a persistent projected CVOL directory: {path}")
    identities = {}
    for name in _CVOL_FILES:
        member = path / name
        if not member.is_file() or member.resolve().parent != path.resolve():
            raise ValueError(f"Invalid member of projected CVOL: {member}")
        identities[name] = _file_identity(member)
    return identities


def _store_ref(ref, work):
    if ref is None:
        return None
    if ref.live_array is not None:
        raise ValueError("A reusable view requires a closed, persistent projected layer")
    location = ref.path.resolve().relative_to(work)
    values = {field.name: getattr(ref, field.name) for field in fields(ref)
              if field.name not in ("path", "live_array")}
    return {"path": location.as_posix(), "files": _store_identity(ref.path), "metadata": values}


def _load_ref(record, work):
    from .interpolation import NrrdLayerRef
    if record is None:
        return None
    name = Path(record["path"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Projected layer path escapes the sweep workspace")
    path = (work / name).resolve()
    path.relative_to(work)
    if _store_identity(path) != record["files"]:
        raise RuntimeError(f"Previously projected cutoff layer changed: {path}")
    values = dict(record["metadata"])
    for key in ("shape", "segment_extent_ijk", "segment_extent_shape_tyx", "sam_group_ids",
                "sam_run_ids", "observation_roots"):
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])
    return NrrdLayerRef(path=path, **values)


class _StorageFileLock:
    """Serialize read-modify-write of fanout_storage.json across scalar child processes.

    Concurrent later-cut children retire their own projected stores and update
    the shared family counters. An exclusive companion lock file works on every
    filesystem the sweep supports, including Windows and NFS without flock. The
    critical section only reads and writes one small JSON file, so a lock older
    than ``stale_seconds`` belongs to a killed owner; recovery starts well
    before a waiter's ``timeout``. These counters are diagnostics; outputs never
    depend on them.
    """

    def __init__(self, path, *, timeout=600.0, stale_seconds=60.0):
        self.path = Path(str(path) + ".lock")
        self.timeout = float(timeout)
        self.stale_seconds = min(float(stale_seconds), self.timeout / 4)
        self._fd = None

    def __enter__(self):
        import time
        deadline = time.monotonic() + self.timeout
        delay = 0.005

        def backoff():
            nonlocal delay
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for sweep storage lock {self.path}")
            time.sleep(delay)
            delay = min(0.1, delay * 2)

        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except (FileExistsError, PermissionError) as error:
                # Windows reports a lock file in its delete-pending state as
                # access denied rather than as existing; both mean "held".
                if isinstance(error, PermissionError) and os.name != "nt":
                    raise
                try:
                    age = time.time() - self.path.stat().st_mtime
                except (FileNotFoundError, PermissionError):
                    backoff()
                    continue
                if age > self.stale_seconds:
                    try:
                        self.path.unlink()
                    except (FileNotFoundError, PermissionError):
                        pass  # gone already, or a live owner holds it open on Windows
                    continue
                backoff()
                continue
            try:
                os.write(fd, str(os.getpid()).encode("ascii") + b"\n")
            except BaseException:
                os.close(fd)
                try:
                    self.path.unlink()
                except OSError:
                    pass
                raise
            self._fd = fd
            return self

    def __exit__(self, *_):
        try:
            os.close(self._fd)
        finally:
            self._fd = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _record_publication_stages(view, metrics, confidence_thread_seconds, layers_seconds):
    """Telemetry: where one sibling publication spent its time."""
    try:
        from .runtime import runtime_telemetry
        telemetry = runtime_telemetry()
        family = getattr(view, "family", "unknown")
        for stage, seconds in (("rank_input", metrics.get("rank_input_seconds", 0.0)),
                               ("projection", metrics.get("projection_seconds", 0.0)),
                               ("confidence_thread", confidence_thread_seconds),
                               ("layers_barrier", layers_seconds)):
            telemetry.add(f"sweep.publication.stage.{stage}_seconds", float(seconds))
            telemetry.add(f"sweep.publication.stage.{family}.{stage}_seconds", float(seconds))
    except Exception:
        pass


def _retire_store(path, expected, parent, cutoff):
    """Remove only this owner's three closed files at their original identities."""
    if (path.name != f"conf_{cutoff:03d}.cvol" or not path.resolve().is_relative_to(parent)
            or not path.is_dir() or {p.name for p in path.iterdir()} != set(_CVOL_FILES)
            or _store_identity(path) != expected):
        return 0
    for name in _CVOL_FILES:
        (path / name).unlink()
    path.rmdir()
    return sum(identity[2] for identity in expected.values())


class ViewMajorPublication:
    """Keep native/LQ writers, branch confidence and transient view receipts aligned."""

    def __init__(self, *, family_root, workspace, sweep_id, cutoffs_hundredths, current_cutoff,
                 scalar_sink, source_shape_tyx, model_name, task, min_conf,
                 confidence_floor_by_backend, progress=None, memory_bytes=512*1024**2):
        self.cutoffs = tuple(_cent(c) for c in cutoffs_hundredths)
        if len(self.cutoffs) < 2 or self.cutoffs != tuple(sorted(set(self.cutoffs))):
            raise ValueError("View-major publication needs multiple distinct increasing cutoffs")
        self.current = _cent(current_cutoff)
        if self.current not in self.cutoffs:
            raise ValueError("Scalar cutoff is absent from the sweep")
        self.sweep_id = sweep_id
        self.run_id = branch_run_id(sweep_id, self.current)
        self.family_root = Path(family_root).resolve()
        self.workspace = Path(workspace).resolve()
        self.work = self.workspace / "view_publication"
        self.model_name = str(model_name)
        self.shape = tuple(map(index, source_shape_tyx))
        self.scalar_sink = scalar_sink
        if scalar_sink.run_id != self.run_id or scalar_sink.output_shape != self.shape:
            raise ValueError("View-major writer and scalar sink disagree on run ID or geometry")
        if scalar_sink.nrrd_dir.resolve() != self._branch_dir(self.current) / "nrrd":
            raise ValueError("Scalar output directory does not belong to this cutoff")
        from .sweep_view_schedule import sweep_publication_concurrency
        self.publishers, self.projection_lanes = sweep_publication_concurrency()
        # Each concurrent publication funds its own CPU lanes and queued planes.
        self.memory_bytes = max(index(memory_bytes), self.projection_lanes * 128 * 1024**2)
        if self.memory_bytes <= 0:
            raise ValueError("Projection memory must be positive")
        self.progress = progress
        self._lock = threading.Lock()
        self._executor = None
        self._confidence_pool = None
        self._layers = None
        self._confidence = {}
        self._floor_views = set()
        self._inflight_views = set()
        self._capacity_probe = {}
        self._adopted_files = {}
        self.geometry_fallbacks = Counter()
        self._closed = False
        self.storage_path = self.family_root / "fanout_storage.json"
        cap = os.environ.get("XTA_YOLO_SWEEP_PROJECTED_GIB")
        self._storage_cap = None
        if cap is not None:
            gib = float(cap)
            if not math.isfinite(gib) or gib < 0:
                raise ValueError("XTA_YOLO_SWEEP_PROJECTED_GIB must be finite and nonnegative")
            self._storage_cap = int(gib * 1024**3)
        self._storage = {"schema": "xta.view_major_storage/1", "sweep_id": self.sweep_id,
                         "scope": "private projected CVOL payloads, indices and metadata",
                         "configured_cap_bytes": self._storage_cap, "live_bytes": 0,
                         "peak_live_bytes": 0, "published_bytes": 0, "retired_bytes": 0,
                         "peak_native_rank_work_bytes": 0, "published_views": 0}
        # Publishers start concurrently; create shared parents first (on
        # Windows, resolving a path while a sibling creates its parent can
        # return an extended-length path that no longer relates to self.work).
        for name in ("projected", "published"):
            (self.work / name).mkdir(parents=True, exist_ok=True)
        self._storage_lock_path = self.workspace / "fanout_storage.json"
        with _StorageFileLock(self._storage_lock_path):
            self._storage = self._read_storage()
        if self.current == self.cutoffs[0]:
            self._start_writers(task, min_conf, confidence_floor_by_backend)
            if self.progress is not None:
                self.scalar_sink.set_layer_publication_observer(self._floor_layer_written)

    def _read_storage(self):
        """Return the shared counters; call with the storage file lock held."""
        current = dict(self._storage)
        if self.storage_path.is_file():
            stored = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if stored.get("sweep_id") != self.sweep_id or stored.get("schema") != current["schema"]:
                raise ValueError("Projection storage counters belong to a different sweep")
            current.update(stored)
        return current

    def _update_storage(self, mutate):
        """Apply one counter delta to the family file under the cross-process lock.

        Call with self._lock held. Later-cut children run concurrently and each
        retires its own stores, so a cached copy must never overwrite the file.
        """
        with _StorageFileLock(self._storage_lock_path):
            current = self._read_storage()
            result = mutate(current)
            current.update(self._capacity_probe)
            write_json_atomic(self.storage_path, current)
            self._storage = current
        return result

    def _floor_layer_written(self, ref):
        if (ref.source == "fullframe" and ref.mask_kind == "yolo" and ref.stage == "pre_interpolation"):
            key = f"{ref.model_name}/{ref.view_name}"
            self.progress.view_projected(key, cutoffs=(self.current,), geometry_passes=1)
            self.progress.view_published(key, self.current, layers=1)

    def finish_floor(self):
        """After all floor views retire, account for the ordinary empty-view no-file decisions."""
        if self.progress is not None and self.current == self.cutoffs[0]:
            self.scalar_sink.wait()
            for key in self._floor_views:
                self.progress.view_projected(key, cutoffs=(self.current,), geometry_passes=0)
                self.progress.view_published(key, self.current, layers=0)

    def _branch_dir(self, cutoff):
        return self.family_root / "confidence_cutoffs" / f"conf_{cutoff:03d}"

    def _start_writers(self, task, min_conf, floor_by_backend):
        from .outputs import NrrdLayerSink
        from .workspace import _cpu_count
        from .sweep_confidence import SweepObservedConfidenceWriter
        from .tta_sweep_branch_layers import TtaSweepBranchLayers
        # Every sibling cutoff of every concurrently published view shares this
        # encoder pool. Each publication fences its own siblings, so no more
        # than publishers x siblings layers are ever queued; a third of the
        # CPUs bounds encoding beside projection lanes and the floor pipeline.
        siblings_count = len(self.cutoffs) - 1
        fanout = max(siblings_count, min(self.publishers * siblings_count,
                                         max(1, int(_cpu_count()) // 3)))
        self._executor = ThreadPoolExecutor(max_workers=fanout, thread_name_prefix="sweep-nrrd")
        # Evidence encoding is ~18% GIL-serialized and stops scaling near 16
        # threads; wider pools only add per-task workspace planes.
        self._confidence_pool = ThreadPoolExecutor(
            max_workers=max(1, min(self.publishers * siblings_count, max(4, int(_cpu_count()) // 8))),
            thread_name_prefix="sweep-confidence")
        siblings = {}
        try:
            for cent in self.cutoffs[1:]:
                root = self._branch_dir(cent)
                uid = branch_run_id(self.sweep_id, cent)
                siblings[cent] = NrrdLayerSink(
                    nrrd_dir=root/"nrrd", stem=self.scalar_sink.stem, run_id=uid,
                    output_shape_tyx=self.shape, max_workers=fanout,
                    low_quality_specs=self.scalar_sink.low_quality_specs,
                    low_quality_root=root/"low_quality", executor=self._executor)
                floors = {backend: max(cent/100., float(floor))
                          for backend, floor in floor_by_backend.items()}
                self._confidence[cent] = SweepObservedConfidenceWriter(
                    output_dir=root, cutoff_hundredths=cent, source_shape_tyx=self.shape,
                    run_id=uid, task=task, min_conf=min_conf, confidence_floor_by_backend=floors)
            self._layers = TtaSweepBranchLayers(siblings, primary_cutoff=self.current,
                                                 max_pending_layers=len(siblings))
        except BaseException:
            self._executor.shutdown(wait=True)
            if self._confidence_pool is not None:
                self._confidence_pool.shutdown(wait=True)
            raise

    def _key(self, view):
        return _digest([self.model_name, view.name])

    def _path(self, cent, view):
        return self.work / "published" / f"conf_{cent:03d}" / (self._key(view)+".json")

    def has_published_view(self, view):
        return self._path(self.current, view).is_file()

    def _available_projection_bytes(self):
        """Call with self._lock held; the allowance is shared by in-flight views."""
        self.work.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.work).free
        # Leave headroom for native producers, filesystem metadata and sibling
        # publications. Recheck for each physical view, never for each slice.
        reserve = min(8 * 1024**3, free // 20)
        available = free - reserve
        if self._storage_cap is not None:
            available = min(available, max(0, self._storage_cap - self._storage["live_bytes"]))
        self._capacity_probe = dict(last_free_bytes=free, last_reserve_bytes=reserve,
                                    last_available_projection_bytes=available)
        self._storage.update(self._capacity_probe)
        # Concurrent publications draw on the same free space: give each of the
        # possible publishers an equal share instead of over-committing it.
        return max(0, available) // max(1, self.publishers, len(self._inflight_views))

    def _record_fallback(self, reason):
        with self._lock:
            self.geometry_fallbacks[reason] += 1
            fallbacks = dict(self.geometry_fallbacks)
            self._update_storage(lambda storage: storage.__setitem__("fallbacks", fallbacks))

    def _receipt_identity(self, cent, view, geometry_digest):
        return {"schema":"xta.prepublished_scalar_view/1", "sweep_id":self.sweep_id,
                "run_id":branch_run_id(self.sweep_id, cent), "cutoff_cent":cent,
                "model_name":self.model_name, "view_name":view.name,
                "source_shape_tyx":list(self.shape), "view_sha256":geometry_digest}

    def publish_view_family(self, view, *, native_shape, read_mask_plane, read_score_plane,
                            counts_by_cutoff, primary_already_published=True,
                            read_mask_crops=None, read_native_crops=None,
                            retain_empty_orthogonal=False):
        """Return sibling refs after durable publication, or None for pre-write fallback.

        Safe to call concurrently for different views.
        """
        if not primary_already_published:
            raise ValueError("The primary scalar owner must retain its own publication")
        siblings = self.cutoffs[1:]
        key = f"{self.model_name}/{view.name}"
        with self._lock:
            if self._closed or self._layers is None:
                raise RuntimeError("Sibling outputs can only be written by the live sweep producer")
            counts = {cent: index(counts_by_cutoff[cent]) for cent in siblings}
            if any(c<0 for c in counts.values()):
                raise ValueError("Invalid native detector count")
            if key in self._inflight_views or any(self._path(cent,view).exists() for cent in siblings):
                raise RuntimeError(f"Cutoff outputs already exist for {view.name}")
            self._inflight_views.add(key)
            self._floor_views.add(key)
            allowance = self._available_projection_bytes()
        try:
            # Views are independent: each owns its projection scope, CVOL stores,
            # confidence layer and receipt paths. Only bookkeeping is serialized,
            # so several publishers can keep pace with inference.
            return self._publish_unlocked(view, key, siblings, counts, allowance,
                native_shape=native_shape, read_mask_plane=read_mask_plane,
                read_score_plane=read_score_plane, read_mask_crops=read_mask_crops,
                read_native_crops=read_native_crops, retain_empty_orthogonal=retain_empty_orthogonal)
        finally:
            with self._lock:
                self._inflight_views.discard(key)

    def _publish_unlocked(self, view, key, siblings, counts, allowance, *, native_shape,
                          read_mask_plane, read_score_plane, read_mask_crops, read_native_crops,
                          retain_empty_orthogonal):
        from .assembly import project_sweep_view
        from .geometry import view_output_token
        from .sweep_backprojection import SweepProjectionUnsupported
        scope = self.work/"projected"/self._key(view)
        metrics = {}
        try:
            refs = project_sweep_view(
                view=view, model_name=self.model_name, cutoffs_hundredths=siblings,
                native_shape=native_shape, output_shape=self.shape, read_mask_plane=read_mask_plane,
                read_mask_crops=read_mask_crops, work_dir=scope,
                workers=self.projection_lanes, memory_bytes=self.memory_bytes,
                metrics=metrics, retain_empty_orthogonal=retain_empty_orthogonal,
                max_workspace_bytes=allowance)
        except SweepProjectionUnsupported as error:
            self._record_fallback(str(error))
            return None
        if set(refs) != set(siblings):
            raise RuntimeError("Projector did not return the entire cutoff family")
        ref_records = {cent: _store_ref(refs[cent], self.work) for cent in siblings}
        geometry_digest = _digest(asdict(view))
        retained = sum(identity[2] for record in ref_records.values() if record is not None
                       for identity in record["files"].values())

        def admit(storage):
            if self._storage_cap is not None and storage["live_bytes"] + retained > self._storage_cap:
                return False
            storage["live_bytes"] += retained
            storage["published_bytes"] += retained
            storage["peak_live_bytes"] = max(storage["peak_live_bytes"], storage["live_bytes"])
            storage["peak_native_rank_work_bytes"] = max(storage["peak_native_rank_work_bytes"],
                                                         metrics["native_rank_backing_bytes"])
            storage["published_views"] += 1
            return True

        with self._lock:
            admitted = self._update_storage(admit)
        if not admitted:
            # Small stores can have more metadata than payload. Check the
            # encoded total before committing any branch rows or evidence.
            for cent, record in ref_records.items():
                if record is not None:
                    if not _retire_store(refs[cent].path, record["files"], scope, cent):
                        raise RuntimeError("Projected store changed before capacity fallback")
            self._record_fallback("Encoded projected stores exceed the requested family limit")
            return None
        # A disjoint family can take multiple nested chains. Record executed passes.
        if self.progress is not None:
            self.progress.view_projected(key, cutoffs=siblings, geometry_passes=metrics["geometry_passes"],
                                         projectors=metrics.get("projectors", ()))
        # Each cutoff's observed evidence runs beside the NRRD encodes (they read
        # different products) on its own native reader; projection has finished
        # with them. A receipt still waits for its cutoff's durable evidence.
        confidence_seconds = []

        def publish_confidence(cent):
            started = time.perf_counter()
            crop_reader = partial(read_native_crops, cent) if read_native_crops is not None else None
            try:
                return self._confidence[cent].publish_view(
                    view, self.model_name, native_shape=native_shape,
                    read_scores=partial(read_score_plane,cent), read_mask=partial(read_mask_plane,cent),
                    read_native_crops=crop_reader)
            finally:
                confidence_seconds.append(time.perf_counter() - started)

        evidence = {cent: self._confidence_pool.submit(publish_confidence, cent) for cent in siblings}
        published = {}

        def completed(cent, _, records):
            # Only record here: evidence errors must not poison the shared
            # NRRD barrier owner, which reserves its failure state for sinks.
            if len(records) != int(refs[cent] is not None):
                raise RuntimeError("Scalar file publication does not match its projected ref")
            published[cent] = records
        layers_started = time.perf_counter()
        try:
            self._layers.publish_view(key,refs,view_token=view_output_token(view),completed=completed)
        finally:
            # Never leave evidence writers running past this view's outcome.
            futures_wait(tuple(evidence.values()))
            _record_publication_stages(view, metrics, sum(confidence_seconds),
                                       time.perf_counter() - layers_started)
        for cent in siblings:
            evidence[cent].result()  # durable evidence precedes every receipt
        for cent in siblings:
            records = published[cent]
            receipt = self._receipt_identity(cent,view,geometry_digest)
            receipt.update(prediction_count=counts[cent], ref=ref_records[cent],
                           layer=records[0] if records else None,
                           geometry_passes=metrics["geometry_passes"])
            path = self._path(cent,view)
            path.parent.mkdir(parents=True,exist_ok=True)
            write_json_atomic(path,receipt)
            if self.progress is not None:
                self.progress.view_published(key,cent,layers=len(records))
        return refs

    def adopt_view(self, view):
        """Attach existing scalar outputs and their source-space ref to this scalar tail."""
        with self._lock:
            if self._closed or self.current == self.cutoffs[0]:
                raise RuntimeError("View adoption requires a live later-cut scalar writer")
            receipt = json.loads(self._path(self.current,view).read_text(encoding="utf-8"))
            expected = self._receipt_identity(self.current,view,_digest(asdict(view)))
            if any(receipt.get(k)!=v for k,v in expected.items()):
                raise ValueError("Published view identity differs from this scalar child")
            ref = _load_ref(receipt["ref"],self.work)
            if ref is not None and (ref.model_name!=self.model_name or ref.view_name!=view.name
                                    or ref.mask_kind!="yolo" or ref.stage!="pre_interpolation"):
                raise ValueError("Projected layer semantics differ from this view")
            count = index(receipt["prediction_count"])
            if count < 0:
                raise ValueError("Invalid published detector count")
            self.scalar_sink.adopt_published_layer(ref,receipt["layer"])
            if ref is not None:
                self._adopted_files[ref.path] = receipt["ref"]["files"]
            return ref,count

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._layers is not None:
                    self._layers.shutdown()
            finally:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                if self._confidence_pool is not None:
                    self._confidence_pool.shutdown(wait=True)
                if self.progress is not None and self.current == self.cutoffs[0]:
                    self.scalar_sink.set_layer_publication_observer(None)
                # Resource teardown joins the scalar sink, union workers and all
                # output readers before closing this owner. Only its own cutoff
                # files are retired; the family owner removes the final directory.
                retired = 0
                for path, identity in self._adopted_files.items():
                    retired += _retire_store(path, identity, self.work / "projected", self.current)
                if retired:
                    def retire(storage):
                        storage["live_bytes"] -= retired
                        storage["retired_bytes"] += retired
                    self._update_storage(retire)
                self._adopted_files.clear()

