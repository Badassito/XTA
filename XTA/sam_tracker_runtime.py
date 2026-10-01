"""Isolated, persistent mask-conditioned tracker for SAM interpolation.

Each endpoint receives a fresh one-object session. Only small artifact handles
cross the worker queue; the predictor and CUDA precision context remain owned
by the admitted worker for its lifetime. Quality selection happens elsewhere.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
import operator
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _image_cache_summary(payload: Mapping[str, object]) -> dict[str, object]:
    """Keep exact demand identity without copying its inventory into every run."""
    records = payload.get("frame_crops", ())
    summary = {key: value for key, value in payload.items() if key != "frame_crops"}
    summary.update(
        storage_format="compact_frame_crops" if records else "full_tyx",
        frame_crops_count=len(records),
        frame_crops_sha256=hashlib.sha256(json.dumps(
            records, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")).hexdigest(),
        descriptor_role="immutable_image_reference_summary",
    )
    return summary


def _request_metadata(value: object | None) -> dict[str, object]:
    """Canonical bounded attribution; arrays and live tracker objects stay out."""
    from .lta_workers import _primitive_mapping

    primitive = _primitive_mapping({} if value is None else value, name="SAM request metadata")
    encoded = json.dumps(primitive, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise ValueError("SAM request metadata exceeds the 64 KiB attribution limit")
    return json.loads(encoded)


def materialize_interpolation_image_cache(
    frames_u8: object,
    *,
    path: Path,
    physical_view_id: str,
    source_identity: str,
):
    """Persist exact native mask-canvas intensities, without a new transform.

    The caller must render the same view/angle/scale canvas used by detector
    observations before using this helper. Gray uint8 input is expanded to RGB
    only when a bounded crop is read inside the tracker worker.
    """

    from .lta_rendering import reference_existing_physical_view_cache

    frames = np.asarray(frames_u8)
    if frames.dtype != np.uint8 or frames.ndim != 3 or min(frames.shape) < 1:
        raise ValueError("SAM image cache must be uint8 TYX on the observation canvas")
    destination = Path(path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            # Write framewise to bound copying for noncontiguous views.
            for frame in frames:
                handle.write(np.ascontiguousarray(frame).tobytes())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return reference_existing_physical_view_cache(
        destination, shape=frames.shape, physical_view_id=physical_view_id,
        source_identity=source_identity,
    )


@dataclass(frozen=True)
class SamTrackerRunResult:
    frames: Mapping[int, np.ndarray]
    tracker_scores: Mapping[int, float | None]
    observation_status: Mapping[int, str]
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class _SubmittedRun:
    input_index: int
    task: Any
    output_directory: Path
    affinity_key: tuple[object, ...]


def execute_interpolation_tracker_task(
    context: object, kind: str, payload: Mapping[str, object],
) -> Mapping[str, object]:
    """Worker entry point: run raw observations and publish a packed artifact."""

    from .lta_experimental import run_mask_seed_session
    from .lta_outputs import write_json_atomically
    from .lta_rendering import LtaPhysicalViewCacheRef, render_native_tile_window
    from .lta_sam import SamSessionPlan
    from .lta_worker_adapter import _jsonable

    worker_started = time.monotonic()
    if kind != "sam_interpolation_run":
        raise ValueError(f"unsupported SAM interpolation task {kind!r}")
    cache_ref = LtaPhysicalViewCacheRef.from_payload(payload["image_cache"])
    cache_ref.revalidate()
    seed_path = Path(str(payload["seed_path"])).resolve(strict=True)
    if _sha256(seed_path) != str(payload["seed_sha256"]):
        raise RuntimeError("SAM interpolation seed artifact checksum changed")
    with np.load(seed_path, allow_pickle=False) as saved:
        seed = np.asarray(saved["seed"], dtype=np.bool_).copy()
    crop = tuple(_integer(value, "crop coordinate") for value in payload["crop_xyxy"])
    if len(crop) != 4:
        raise ValueError("SAM crop requires four XYXY coordinates")
    x0, y0, x1, y1 = crop
    shape = (y1 - y0, x1 - x0)
    if seed.shape != shape or not bool(seed.any()):
        raise ValueError("SAM seed must be nonempty and match its declared crop geometry")
    start = _integer(payload["frame_start"], "frame_start")
    stop = _integer(payload["frame_stop"], "frame_stop")
    prompt = _integer(payload["seed_frame"], "seed_frame")
    direction = str(payload["direction"])
    if direction not in {"forward", "backward"}:
        raise ValueError("SAM endpoint sessions require forward or backward propagation")
    if not start <= prompt < stop:
        raise ValueError("SAM seed frame lies outside the bounded interval")
    session = SamSessionPlan(
        sequence_id=str(payload["run_id"]), session_index=0,
        frame_start=start, frame_stop=stop,
    )
    render_started = time.perf_counter()
    resource = render_native_tile_window(
        cache_ref, frame_start=start, frame_stop=stop, tile_xyxy=crop,
    )
    render_seconds = time.perf_counter() - render_started
    feature_cache = getattr(context, "feature_cache", None)
    cache_before = None if feature_cache is None else feature_cache.snapshot()
    observations: dict[int, object] = {}

    def observe(item: object) -> None:
        frame = int(item.frame_index)
        if int(item.object_id) != 0 or frame in observations:
            raise RuntimeError("SAM raw tracker returned an unexpected or duplicate object/frame")
        if tuple(item.binary_mask.shape) != shape:
            raise RuntimeError("SAM raw tracker output changed crop coordinates")
        observations[frame] = item

    tracker_started = time.perf_counter()
    result = run_mask_seed_session(
        context.predictor, context.predictor, resource=resource, session=session,
        prompt_frame=prompt, ground_truth=seed, seed=None, object_masks=(seed,),
        conf=0.0, propagation_mode="tracker-only", empty_frame_limit=None,
        propagation_direction=direction, retain_predictions=False,
        raw_observation_callback=observe, retain_raw_observations=False,
        seed_roundtrip_policy="exact",
        feature_cache=feature_cache,
        cache_frame_identity=lambda local: (
            cache_ref.identity_sha256, cache_ref.physical_view_id, crop,
            int(start) + int(local), "pinned_rgb_u8_loader_v1",
        ),
    )
    tracker_seconds = time.perf_counter() - tracker_started
    expected = (
        tuple(range(prompt, stop)) if direction == "forward"
        else tuple(range(start, prompt + 1))
    )
    if tuple(sorted(observations)) != expected or not result["raw_observation_complete"]:
        raise RuntimeError("SAM raw tracker did not observe every expected frame including terminal")
    if not bool(result["seed_roundtrip_passed"]):
        raise RuntimeError("SAM tracker mask injection failed its geometry contract")
    frames = np.stack([observations[frame].binary_mask for frame in expected])
    scores = np.asarray([
        np.nan if observations[frame].frame_tracker_score is None
        else float(observations[frame].frame_tracker_score)
        for frame in expected
    ], dtype=np.float64)
    removed = np.asarray([
        observations[frame].status == "removed" for frame in expected
    ], dtype=np.bool_)
    output_dir = Path(str(payload["output_dir"])).resolve(strict=False)
    mask_path = output_dir / "raw_masks.npz"
    pack_started = time.perf_counter()
    _atomic_npz(
        mask_path, packed_masks=np.packbits(frames.reshape(len(expected), -1), axis=1),
        frame_indices=np.asarray(expected, dtype=np.int64),
        shape=np.asarray(shape, dtype=np.int64), tracker_scores=scores, removed=removed,
    )
    # Remove retained previews and compatibility fields containing mask arrays;
    # evidence confidence and detector lineage remain separate in the caller.
    receipt_keys = (
        "raw_observation_count", "raw_observation_callback_count", "raw_observation_boundary",
        "raw_observation_complete", "model_visited_frame_ranges", "policy_zero_frame_ranges",
        "seed_roundtrip_passed", "seed_roundtrip_exact", "seed_object_metrics",
        "anchor_integrity_passed", "anchor_integrity_object_metrics",
        "frame_tracker_score_semantics", "removed_object_observations",
        "tracker_feature_preparation", "propagation_direction",
    )
    receipt = {key: result[key] for key in receipt_keys}
    manifest = {
        "schema": "xta.sam-raw-tracker-run/1", "run_id": str(payload["run_id"]),
        "status": "invalid_removed_object" if bool(removed.any()) else "complete",
        "prediction_valid": not bool(removed.any()),
        "coverage_complete": True, "crop_xyxy": list(crop), "seed_frame": prompt,
        "seed_artifact_sha256": str(payload["seed_sha256"]),
        "frame_range": [start, stop], "expected_frames": list(expected),
        "direction": direction, "image_cache": _image_cache_summary(cache_ref.payload()),
        "raw_masks": {"path": str(mask_path), "sha256": _sha256(mask_path),
                      "size_bytes": mask_path.stat().st_size},
        "adapter_receipt": _jsonable(receipt),
        "profile": _jsonable(context.profile), "sam_runtime": _jsonable(context.sam_runtime),
        "sam_model": _jsonable(getattr(context, "sam_model", {})),
        "request_metadata": _request_metadata(payload.get("request_metadata")),
        "feature_cache_before": _jsonable(cache_before),
        "feature_cache_after": _jsonable(None if feature_cache is None else feature_cache.snapshot()),
        "temporary_artifact_directory": str(output_dir),
        "timings": {
            "worker_started_monotonic": worker_started,
            "queue_seconds": max(0.0, worker_started - float(payload.get("submitted_monotonic", worker_started))),
            "render_seconds": render_seconds, "tracker_seconds": tracker_seconds,
            "pack_seconds": time.perf_counter() - pack_started,
            "worker_seconds": time.monotonic() - worker_started,
        },
        "seed_policy": "one original endpoint; held-out terminal; no hole filling",
        "detector_confidence": None,
    }
    manifest_path = write_json_atomically(output_dir / "manifest.json", manifest)
    return {
        "artifact_path": str(manifest_path),
        "metrics": {"frame_count": len(expected), "foreground_pixels": int(frames.sum())},
        "metadata": {"raw": True, "status": str(manifest["status"])},
    }


def load_tracker_run_result(
    manifest_path: Path, *, expected_sha256: str | None = None,
) -> SamTrackerRunResult:
    """Verify file-backed transfer and unpack complete attributable observations."""

    path = Path(manifest_path).resolve(strict=True)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise RuntimeError("SAM tracker manifest checksum changed in transfer")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "xta.sam-raw-tracker-run/1":
        raise RuntimeError("SAM tracker returned an unsupported evidence schema")
    packet = manifest["raw_masks"]
    artifact = Path(packet["path"]).resolve(strict=True)
    if artifact.parent != path.parent:
        raise RuntimeError("SAM raw masks artifact escaped its run directory")
    if artifact.stat().st_size != packet["size_bytes"] or _sha256(artifact) != packet["sha256"]:
        raise RuntimeError("SAM raw mask transfer checksum/size changed")
    with np.load(artifact, allow_pickle=False) as saved:
        packed = saved["packed_masks"].copy()
        if saved["shape"].dtype.kind not in "iu" or saved["frame_indices"].dtype.kind not in "iu":
            raise RuntimeError("SAM raw frame/shape packet requires integer coordinates")
        shape = tuple(_integer(value, "raw shape") for value in saved["shape"])
        frames = tuple(_integer(value, "raw frame") for value in saved["frame_indices"])
        scores = saved["tracker_scores"].copy()
        removed = saved["removed"].copy()
    crop = tuple(_integer(value, "manifest crop") for value in manifest["crop_xyxy"])
    expected_shape = (crop[3] - crop[1], crop[2] - crop[0])
    if shape != expected_shape or min(shape) < 1:
        raise RuntimeError("SAM raw mask shape differs from its declared crop")
    count = len(frames)
    start, stop = (_integer(value, "manifest frame range") for value in manifest["frame_range"])
    prompt = _integer(manifest["seed_frame"], "manifest seed frame")
    direction = str(manifest["direction"])
    if not 0 <= start <= prompt < stop or direction not in {"forward", "backward"}:
        raise RuntimeError("SAM raw packet has an invalid directed interval")
    expected_frames = (
        tuple(range(prompt, stop)) if direction == "forward" else tuple(range(start, prompt + 1))
    )
    if frames != expected_frames or frames != tuple(manifest["expected_frames"]) or len(set(frames)) != count:
        raise RuntimeError("SAM raw frame packet has missing or duplicate observations")
    if scores.shape != (count,) or removed.shape != (count,) or removed.dtype != np.bool_:
        raise RuntimeError("SAM raw score/status packet has malformed shape")
    if packed.dtype != np.uint8 or packed.shape != (count, (math.prod(shape) + 7) // 8):
        raise RuntimeError("SAM raw mask packet has malformed packed geometry")
    if not manifest.get("coverage_complete") or bool(manifest.get("prediction_valid")) != (not bool(removed.any())):
        raise RuntimeError("SAM raw packet validity/coverage disagrees with observation status")
    if not all(
        bool(flag) and math.isnan(float(score)) or
        not bool(flag) and math.isfinite(float(score)) and 0.0 <= float(score) <= 1.0
        for flag, score in zip(removed, scores)
    ):
        raise RuntimeError("SAM raw tracker score/status packet has invalid values")
    masks = np.unpackbits(packed, axis=1, count=math.prod(shape)).reshape(count, *shape).astype(bool)
    masks.setflags(write=False)
    return SamTrackerRunResult(
        frames={frame: masks[index] for index, frame in enumerate(frames)},
        tracker_scores={frame: None if removed[index] else float(scores[index])
                        for index, frame in enumerate(frames)},
        observation_status={frame: "removed" if removed[index] else "observed"
                            for index, frame in enumerate(frames)},
        receipt=manifest,
    )


class SamInterpolationTracker:
    """One admitted pool reused across endpoint runs and provenance scopes.

    GPU admission belongs to the main pipeline. Constructing this object is
    accelerator-light; call start only after detector residency is retired and
    keep the corresponding GPU lease until close completes.
    """

    def __init__(
        self, *, model_path: str | Path, device_ids: Sequence[int],
        artifact_root: str | Path, source_cache_ref: object | None = None,
        profile: str = "auto", startup_timeout: float = 300.0,
        feature_cache_bytes: int = 512 * 1024 * 1024,
        feature_cache_headroom_bytes: int | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.device_ids = tuple(_integer(value, "device_id") for value in device_ids)
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids) or min(self.device_ids) < 0:
            raise ValueError("SAM tracker requires unique nonnegative CUDA devices")
        self.artifact_root = Path(artifact_root).resolve(strict=False)
        self.profile = str(profile)
        self.startup_timeout = float(startup_timeout)
        self.feature_cache_bytes = _integer(feature_cache_bytes, "feature_cache_bytes")
        self.feature_cache_headroom_bytes = (
            None if feature_cache_headroom_bytes is None
            else _integer(feature_cache_headroom_bytes, "feature_cache_headroom_bytes")
        )
        if self.feature_cache_bytes < 0 or (
            self.feature_cache_headroom_bytes is not None and self.feature_cache_headroom_bytes < 0
        ):
            raise ValueError("SAM feature-cache byte limits must be nonnegative")
        self._pool = None
        self._residency_released = True
        self._closed = False
        self._run_index = 0
        self._cancel = threading.Event()
        self._cancel_reason = "SAM tracker operation cancelled"
        self._dispatch_lock = threading.Lock()
        self._iteration_active = False
        self._crop_affinity: OrderedDict[tuple[object, ...], int] = OrderedDict()
        self._device_submissions = {device: 0 for device in self.device_ids}
        self.dispatch_stats = {
            "submitted": 0, "completed": 0, "peak_in_flight": 0,
            "affinity_hits": 0, "affinity_lends": 0, "wait_seconds": 0.0,
        }
        self._source_cache_ref = None
        if source_cache_ref is not None:
            self.set_source_cache(source_cache_ref)

    def set_source_cache(self, cache_ref: object) -> None:
        from .lta_rendering import LtaPhysicalViewCacheRef

        if not isinstance(cache_ref, LtaPhysicalViewCacheRef):
            raise TypeError("SAM source_cache_ref must be a LtaPhysicalViewCacheRef")
        if self._iteration_active:
            raise RuntimeError("cannot replace SAM source cache while a bounded iterator is active")
        cache_ref.revalidate()
        self._source_cache_ref = cache_ref

    def start(self) -> "SamInterpolationTracker":
        from .lta_workers import LtaWorkerInit, LtaWorkerPool

        if self._closed:
            raise RuntimeError("SAM tracker has been closed")
        if self._cancel.is_set():
            raise RuntimeError(self._cancel_reason)
        if self._pool is None:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            self._residency_released = False
            try:
                self._pool = LtaWorkerPool(
                    self.device_ids,
                    LtaWorkerInit(
                        adapter_module="XTA.sam_tracker_runtime",
                        adapter_factory="build_interpolation_predictor",
                        adapter_execute="execute_interpolation_tracker_task",
                        adapter_shutdown="close_interpolation_predictor",
                        adapter_config={"model_path": self.model_path, "profile": self.profile,
                                        "conf": 0.0, "max_num_objects": 16,
                                        "feature_cache_bytes": self.feature_cache_bytes,
                                        "feature_cache_headroom_bytes": self.feature_cache_headroom_bytes},
                    ), startup_timeout=self.startup_timeout, cancel_event=self._cancel,
                )
            except BaseException as error:
                self._pool = getattr(error, "unsettled_worker_pool", None)
                self._residency_released = self._pool is None
                raise
        return self

    @property
    def workers_settled(self) -> bool:
        return self._residency_released if self._pool is None else bool(self._pool.workers_settled)

    @property
    def residency_released(self) -> bool:
        return self.workers_settled

    def _settle_pool(self, *, timeout: float) -> None:
        pool = self._pool
        if pool is None:
            return
        original_error = None
        try:
            pool.shutdown(timeout=timeout, force=True)
        except BaseException as error:
            original_error = error
        if not bool(pool.workers_settled):
            try:
                pool.force_close(timeout=1.0)
            except BaseException as error:
                if original_error is None:
                    original_error = error
                else:
                    add_note = getattr(original_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"SAM forced worker cleanup also failed: {type(error).__name__}: {error}")
        self._residency_released = bool(pool.workers_settled)
        if self._residency_released:
            self._pool = None
        if original_error is not None:
            raise original_error
        if not self._residency_released:
            raise RuntimeError("SAM model residency is unproven after worker shutdown")

    def _prepare_task(
        self, request: Mapping[str, object], *, cache_ref: object,
        input_index: int, staging_directories: set[Path],
    ) -> _SubmittedRun:
        from .lta_sam import SamSessionPlan
        from .lta_workers import LtaWorkerTask

        if not isinstance(request, Mapping):
            raise TypeError("SAM tracker requests must be mappings")
        run_id = str(request["run_id"]).strip()
        if not run_id:
            raise ValueError("SAM tracker run_id must be nonempty")
        cache_ref.revalidate()
        start = _integer(request["frame_start"], "frame_start")
        stop = _integer(request["frame_stop"], "frame_stop")
        prompt = _integer(request["seed_frame"], "seed_frame")
        direction = str(request["direction"])
        SamSessionPlan(sequence_id=str(run_id), session_index=0, frame_start=start, frame_stop=stop)
        if not start <= prompt < stop or direction not in {"forward", "backward"}:
            raise ValueError("SAM endpoint seed/direction does not fit its bounded session")
        crop = tuple(_integer(value, "crop coordinate") for value in request["crop_xyxy"])
        if len(crop) != 4:
            raise ValueError("SAM context crop must contain four XYXY coordinates")
        x0, y0, x1, y1 = crop
        source_shape = cache_ref.shape
        if not (0 <= x0 < x1 <= source_shape[2] and 0 <= y0 < y1 <= source_shape[1]
                and 0 <= start < stop <= source_shape[0]):
            raise ValueError("SAM context crop/interval is outside the immutable image canvas")
        seed = np.asarray(request["seed_mask"])
        if seed.dtype != np.bool_ or seed.shape != (y1 - y0, x1 - x0) or not bool(seed.any()):
            raise ValueError("SAM original endpoint seed must be boolean nonempty crop-space HxW")
        metadata = _request_metadata(request.get("metadata"))
        token = uuid.uuid4().hex
        output_dir = self.artifact_root / f"run-{self._run_index:06d}-{token}"
        self._run_index += 1
        staging_directories.add(output_dir)
        seed_path = output_dir / "seed.npz"
        _atomic_npz(seed_path, seed=seed)
        task = LtaWorkerTask(
            work_id=str(run_id), attempt_token=token, kind="sam_interpolation_run",
            payload={"run_id": str(run_id), "seed_path": str(seed_path),
                     "seed_sha256": _sha256(seed_path), "image_cache": cache_ref.payload(),
                     "crop_xyxy": list(crop), "seed_frame": prompt, "frame_start": start,
                     "frame_stop": stop, "direction": direction, "output_dir": str(output_dir),
                     "request_metadata": metadata,
                     "input_index": int(input_index), "prepared_monotonic": time.monotonic()},
        )
        return _SubmittedRun(input_index, task, output_dir, (cache_ref.identity_sha256, crop))

    def _remove_staging(self, directory: Path) -> None:
        resolved = Path(directory).resolve(strict=False)
        if resolved.parent != self.artifact_root or not resolved.name.startswith("run-"):
            raise RuntimeError("SAM staging cleanup target is outside the admitted artifact root")
        if resolved.exists():
            shutil.rmtree(resolved)

    def _wait_result(self):
        while True:
            if self._cancel.is_set():
                raise RuntimeError(self._cancel_reason)
            try:
                return self._pool.wait_result(timeout=0.5)
            except TimeoutError:
                continue
            except RuntimeError as error:
                if self._cancel.is_set():
                    raise RuntimeError(self._cancel_reason) from error
                raise

    def iter_results(
        self, requests: Iterable[Mapping[str, object]], *,
        source_cache_ref: object | None = None, max_in_flight: int | None = None,
    ) -> Iterator[tuple[int, SamTrackerRunResult]]:
        """Yield attributable completions with bounded, work-conserving dispatch.

        One endpoint session occupies each admitted worker. A freed worker takes
        another request before its verified result is yielded. At most the
        configured number of jobs and one yielded transfer directory exist;
        advancing the iterator releases that transfer's temporary files. Result
        masks remain independently owned CPU arrays. Stable input indices retain
        attribution when consumers pack evidence in execution completion order.

        Different scope iterators serialize only their GPU execution because the
        worker pool has one result consumer. Each captures an immutable image
        reference, allowing other scopes to plan/render/measure concurrently.
        """

        from .lta_rendering import LtaPhysicalViewCacheRef

        capacity = len(self.device_ids) if max_in_flight is None else _integer(max_in_flight, "max_in_flight")
        if not 1 <= capacity <= len(self.device_ids):
            raise ValueError("max_in_flight must be between one and the admitted worker count")
        while not self._dispatch_lock.acquire(timeout=0.05):
            if self._cancel.is_set() or self._closed:
                raise RuntimeError(self._cancel_reason if self._cancel.is_set() else "SAM tracker has been closed")
        staging_directories: set[Path] = set()
        pending: dict[tuple[str, str], tuple[_SubmittedRun, int]] = {}
        complete = False
        try:
            if self._closed or self._cancel.is_set():
                raise RuntimeError(self._cancel_reason if self._cancel.is_set() else "SAM tracker has been closed")
            cache_ref = source_cache_ref if source_cache_ref is not None else self._source_cache_ref
            if not isinstance(cache_ref, LtaPhysicalViewCacheRef):
                raise RuntimeError("SAM tracker requires an immutable image cache for this iterator")
            cache_ref.revalidate()
            self._iteration_active = True
            iterator = iter(requests)
            source_exhausted = False
            next_index = 0
            seen_run_ids: set[str] = set()
            free_devices = set(self.device_ids)

            def submit_next() -> bool:
                nonlocal source_exhausted, next_index
                if source_exhausted or not free_devices:
                    return False
                if self._cancel.is_set():
                    raise RuntimeError(self._cancel_reason)
                try:
                    request = next(iterator)
                except StopIteration:
                    source_exhausted = True
                    return False
                prepared = self._prepare_task(
                    request, cache_ref=cache_ref, input_index=next_index,
                    staging_directories=staging_directories,
                )
                if prepared.task.work_id in seen_run_ids:
                    raise ValueError(f"duplicate SAM run_id in one iterator: {prepared.task.work_id}")
                seen_run_ids.add(prepared.task.work_id)
                next_index += 1
                self.start()
                # Separate request preparation/model startup from actual queue
                # residence. Primitive task payloads are immutable after submit.
                from .lta_workers import LtaWorkerTask
                payload = dict(prepared.task.payload)
                payload["submitted_monotonic"] = time.monotonic()
                queued_task = LtaWorkerTask(
                    work_id=prepared.task.work_id, attempt_token=prepared.task.attempt_token,
                    kind=prepared.task.kind, payload=payload,
                )
                prepared = _SubmittedRun(prepared.input_index, queued_task,
                                         prepared.output_directory, prepared.affinity_key)
                owner = self._crop_affinity.get(prepared.affinity_key)
                if owner in free_devices:
                    device = owner
                    self.dispatch_stats["affinity_hits"] += 1
                else:
                    device = min(free_devices, key=lambda value: (self._device_submissions[value], value))
                    if owner is not None:
                        self.dispatch_stats["affinity_lends"] += 1
                self._pool.submit(prepared.task, execution_device_id=device)
                free_devices.remove(device)
                self._device_submissions[device] += 1
                self._crop_affinity[prepared.affinity_key] = device
                self._crop_affinity.move_to_end(prepared.affinity_key)
                if len(self._crop_affinity) > 4096:
                    self._crop_affinity.popitem(last=False)
                pending[(prepared.task.work_id, prepared.task.attempt_token)] = (prepared, device)
                self.dispatch_stats["submitted"] += 1
                self.dispatch_stats["peak_in_flight"] = max(self.dispatch_stats["peak_in_flight"], len(pending))
                return True

            while len(pending) < capacity and submit_next():
                pass
            while pending:
                waiting_started = time.perf_counter()
                event = self._wait_result()
                self.dispatch_stats["wait_seconds"] += time.perf_counter() - waiting_started
                identity = event.work_id, event.attempt_token
                if identity not in pending:
                    raise RuntimeError("SAM worker returned a result for another bounded iterator")
                prepared, device = pending.pop(identity)
                if int(event.execution_device_id) != device:
                    raise RuntimeError("SAM worker result changed its admitted device ownership")
                free_devices.add(device)
                # Start the next GPU session before CPU transfer verification and
                # evidence packing; completed and active runs never share masks.
                while len(pending) < capacity and submit_next():
                    pass
                artifact_path = Path(event.artifact_path).resolve(strict=True)
                if artifact_path != prepared.output_directory.resolve(strict=True) / "manifest.json":
                    raise RuntimeError("SAM worker manifest escaped its assigned run staging directory")
                result = load_tracker_run_result(
                    artifact_path, expected_sha256=event.artifact_sha256,
                )
                if str(result.receipt.get("run_id")) != prepared.task.work_id:
                    raise RuntimeError("SAM raw evidence changed its immutable run identity")
                expected_contract = {
                    "crop_xyxy": prepared.task.payload["crop_xyxy"],
                    "frame_range": [prepared.task.payload["frame_start"], prepared.task.payload["frame_stop"]],
                    "seed_frame": prepared.task.payload["seed_frame"],
                    "direction": prepared.task.payload["direction"],
                    "image_cache": _image_cache_summary(prepared.task.payload["image_cache"]),
                    "seed_artifact_sha256": prepared.task.payload["seed_sha256"],
                    "temporary_artifact_directory": str(prepared.output_directory),
                }
                if any(result.receipt.get(key) != value for key, value in expected_contract.items()):
                    raise RuntimeError("SAM raw evidence changed its immutable image/seed/geometry contract")
                if result.receipt.get("request_metadata", {}) != prepared.task.payload["request_metadata"]:
                    raise RuntimeError("SAM raw evidence changed its immutable original-run/tile attribution")
                receipt = dict(result.receipt)
                receipt["dispatch"] = {"input_index": prepared.input_index,
                                       "execution_device_id": device, "worker_pid": event.worker_pid}
                result = SamTrackerRunResult(result.frames, result.tracker_scores, result.observation_status, receipt)
                self.dispatch_stats["completed"] += 1
                yield prepared.input_index, result
                self._remove_staging(prepared.output_directory)
                staging_directories.discard(prepared.output_directory)
                del result
            complete = True
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            workers_settled = True
            if not complete and (pending or self._pool is not None):
                # Only this consumer shuts down queues. Asynchronous cancellation
                # merely sets an event, so no wait-result/shutdown race is possible.
                self.cancel("SAM bounded generation failed or its consumer stopped before completion")
                if self._pool is not None:
                    try:
                        self._settle_pool(timeout=1.0)
                    except BaseException as error:
                        cleanup_error = error
                    workers_settled = self.workers_settled
                self._closed = True
            try:
                if workers_settled:
                    for directory in staging_directories:
                        self._remove_staging(directory)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            finally:
                self._iteration_active = False
                self._dispatch_lock.release()
            if cleanup_error is not None:
                if primary_error is None or isinstance(primary_error, GeneratorExit):
                    raise cleanup_error
                add_note = getattr(primary_error, "add_note", None)
                if callable(add_note):
                    add_note(f"SAM bounded staging cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")

    def run(
        self, *, run_id: str, seed_mask: object, seed_frame: int,
        frame_start: int, frame_stop: int, direction: str,
        crop_xyxy: Sequence[int],
        metadata: Mapping[str, object] | None = None,
    ) -> SamTrackerRunResult:
        """Synchronous compatibility wrapper over the bounded execution path."""

        request = dict(run_id=run_id, seed_mask=seed_mask, seed_frame=seed_frame,
                       frame_start=frame_start, frame_stop=frame_stop,
                       direction=direction, crop_xyxy=crop_xyxy, metadata=metadata)
        result = None
        for _index, result in self.iter_results((request,), max_in_flight=1):
            pass
        if result is None:
            raise RuntimeError("SAM endpoint execution returned no observation evidence")
        return result

    def close(self) -> None:
        try:
            self._settle_pool(timeout=1.0 if self._cancel.is_set() else 30.0)
        finally:
            self._closed = True

    def cancel(self, reason: str = "SAM tracker operation cancelled") -> None:
        """Interrupt parent-side startup/result waits without a queue race.

        The inference-owning thread exits with an infrastructure error. Its
        owner then closes the pool after joining consumers; partial artifacts
        are diagnostic data and never a completed empty support layer.
        """

        self._cancel_reason = str(reason)
        self._cancel.set()

    def release_result(self, result: SamTrackerRunResult) -> None:
        """Delete verified transfer staging once generation has packed evidence.

        The result's independent CPU arrays remain valid. Failed runs are kept
        for diagnostics; successful transfer directories need not accumulate
        with the number of endpoint hypotheses.
        """

        if not isinstance(result, SamTrackerRunResult):
            raise TypeError("result must be a SamTrackerRunResult")
        directory = Path(str(result.receipt["temporary_artifact_directory"])).resolve(strict=False)
        self._remove_staging(directory)

    def __enter__(self) -> "SamInterpolationTracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_interpolation_predictor(config: Mapping[str, object]):
    from .lta_worker_adapter import build_worker_predictor, close_worker_predictor
    from .lta_sam import resolve_local_sam_bundle

    context = build_worker_predictor(config)
    try:
        tracker = context.predictor.model.tracker
        if getattr(tracker, "fill_hole_area", None) != 0:
            raise RuntimeError("raw SAM interpolation requires SDK output hole filling disabled")
        bundle = resolve_local_sam_bundle(str(config["model_path"]))
        context.sam_model = {
            "checkpoint_path": str(bundle.checkpoint_path),
            "checkpoint_sha256": _sha256(bundle.checkpoint_path),
            "checkpoint_identity_sha256": bundle.checkpoint_identity_sha256,
            "model_version": bundle.model_version,
            "bpe_path": None if bundle.bpe_path is None else str(bundle.bpe_path),
            "sdk_output_hole_fill_area": 0,
        }
        requested_cache_bytes = _integer(config.get("feature_cache_bytes", 0), "feature_cache_bytes")
        if requested_cache_bytes > 0:
            from .lta_feature_cache import LruTrackerFeatureCache

            torch = context.torch_module
            _free, total_bytes = torch.cuda.mem_get_info(0)
            requested_headroom = config.get("feature_cache_headroom_bytes")
            mandatory_headroom = max(2 * 1024 * 1024 * 1024, int(total_bytes * 0.15))
            headroom = mandatory_headroom if requested_headroom is None else max(
                mandatory_headroom, _integer(requested_headroom, "feature_cache_headroom_bytes"),
            )

            def memory_probe() -> int:
                free_bytes, _total = torch.cuda.mem_get_info(0)
                # Reusable allocator reserve can satisfy future feature tensors;
                # active allocations and still-live evicted entries remain counted.
                reusable = max(0, int(torch.cuda.memory_reserved(0)) - int(torch.cuda.memory_allocated(0)))
                return int(free_bytes) + reusable

            budget = min(requested_cache_bytes, max(0, memory_probe() - headroom))
            context.feature_cache = LruTrackerFeatureCache(
                max_bytes=budget, memory_probe=memory_probe, headroom_bytes=headroom,
                model_identity=json.dumps({"model": context.sam_model, "runtime": context.sam_runtime,
                                           "profile": context.profile}, sort_keys=True),
                position_source_identity=str(context.sam_runtime.get("package_tree_sha256", "")),
            )
        else:
            context.feature_cache = None
    except BaseException:
        close_worker_predictor(context)
        raise
    return context


def close_interpolation_predictor(context: object) -> None:
    from .lta_worker_adapter import close_worker_predictor

    feature_cache = getattr(context, "feature_cache", None)
    if feature_cache is not None:
        feature_cache.close()
        context.feature_cache = None
    close_worker_predictor(context)
