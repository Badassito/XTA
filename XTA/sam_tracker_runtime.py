"""Isolated, persistent mask-conditioned tracker for SAM interpolation.

Each endpoint receives a fresh one-object session. Only small artifact handles
cross the worker queue; the predictor and CUDA precision context remain owned
by the admitted worker for its lifetime. Quality selection happens elsewhere.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import copy
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
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


_LIVE_SAM_TRACKERS = weakref.WeakSet()
_LIVE_SAM_TRACKERS_LOCK = threading.Lock()
_SAM_SNAPSHOT_COUNTS = ('active_scopes', 'credited_scopes', 'ready_jobs', 'preparing_jobs',
    'running_jobs', 'acked_awaiting_consumer_jobs', 'consumer_held_jobs',
    'live_prepared_bank_bytes', 'submitted_jobs', 'completed_jobs', 'completion_acks')


def _sam_scheduler_sample():
    with _LIVE_SAM_TRACKERS_LOCK:
        trackers = tuple(_LIVE_SAM_TRACKERS)
    snapshots = tuple(tracker.snapshot() for tracker in trackers)
    return dict(sample_monotonic_ns=time.monotonic_ns(), tracker_instances=len(snapshots),
        closed_instances=sum(row['closed'] for row in snapshots),
        cancelled_instances=sum(row['cancelled'] for row in snapshots),
        **{key: sum(row[key] for row in snapshots) for key in _SAM_SNAPSHOT_COUNTS})


def _register_sam_scheduler_sample(tracker):
    """Use existing sampling only; a module provider retains no tracker."""
    try:
        from . import runtime
        telemetry = runtime._RUNTIME_TELEMETRY
        if (telemetry is None or not telemetry.enabled
                or not runtime._env_flag('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', True)):
            return
        with _LIVE_SAM_TRACKERS_LOCK:
            _LIVE_SAM_TRACKERS.add(tracker)
        telemetry.register_sample_provider('sam.scheduler.live', _sam_scheduler_sample)
    except BaseException:
        # Optional diagnostics cannot change tracker success or resource ownership.
        with _LIVE_SAM_TRACKERS_LOCK:
            _LIVE_SAM_TRACKERS.discard(tracker)


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


@dataclass(frozen=True)
class SamTrackerFamily:
    """Sealed job identities and one parent-only lazy seed factory.

    No masks or tracker state are retained by this descriptor. The factory is
    called only on its producer after credit is reserved for its original index.
    """

    family_id: str
    input_indices: tuple[int, ...]
    run_ids: tuple[str, ...]
    request_factory: Callable[[int], Mapping[str, object]]
    frame_work_proxy: int = 0

    def __post_init__(self):
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("SAM family identity must be a nonempty string")
        identity = self.family_id
        indices = tuple(_integer(index, "family input_index") for index in self.input_indices)
        runs = tuple(self.run_ids)
        if not identity or not indices or len(indices) != len(runs):
            raise ValueError("SAM family needs an identity and matching nonempty index/run inventories")
        if min(indices) < 0 or len(set(indices)) != len(indices) or any(not isinstance(value,str) or not value for value in runs):
            raise ValueError("SAM family input indices/run identities must be valid and unique")
        if len(set(runs)) != len(runs) or not callable(self.request_factory):
            raise ValueError("SAM family requires unique run IDs and a callable lazy request factory")
        work = _integer(self.frame_work_proxy, "family frame_work_proxy")
        if work < 0:
            raise ValueError("SAM family work proxy cannot be negative")
        object.__setattr__(self, "family_id", identity)
        object.__setattr__(self, "input_indices", indices)
        object.__setattr__(self, "run_ids", runs)
        object.__setattr__(self, "frame_work_proxy", work)


class _FamilyDispatch:
    """Virtual producer-lane cursors; pending descriptors hold no seed arrays."""

    def __init__(self, families, schedule):
        sealed = tuple(families)
        if any(not isinstance(family, SamTrackerFamily) for family in sealed):
            raise TypeError("Family scheduling requires sealed SamTrackerFamily descriptors")
        identities = [family.family_id for family in sealed]
        indices = [index for family in sealed for index in family.input_indices]
        runs = [run_id for family in sealed for run_id in family.run_ids]
        if len(set(identities)) != len(identities) or len(set(indices)) != len(indices) or len(set(runs)) != len(runs):
            raise ValueError("SAM family schedule contains duplicate family/index/run ownership")
        if schedule not in {"fifo", "lpt"}:
            raise ValueError("Experimental SAM family schedule must be fifo or lpt")
        if schedule == "lpt":
            sealed = tuple(sorted(sealed, key=lambda value: (-value.frame_work_proxy, value.input_indices[0])))
        self.waiting = iter(sealed)
        self.waiting_exhausted = False
        self.active = {}
        self.total_jobs = len(indices)
        self.assigned_families = self.completed_families = self.peak_active = 0
        self.execution_order = []

    def retire_completed(self, free_devices):
        """Retire exhausted preparation cursors, independently of SDK completion.

        Final family receipts still require every actual job to be submitted and
        consumed. Cursor exhaustion alone never certifies a completed family.
        """
        for device in free_devices:
            current = self.active.get(device)
            if current is not None and current[1] == len(current[0].input_indices):
                self.completed_families += 1
                del self.active[device]

    def next_for(self, free_devices):
        for device in sorted(free_devices):
            self.retire_completed((device,))
            current = self.active.get(device)
            if current is None and not self.waiting_exhausted:
                family = next(self.waiting, None)
                if family is None:
                    self.waiting_exhausted = True
                else:
                    current = (family, 0)
                    self.active[device] = current
                    self.assigned_families += 1
                    self.peak_active = max(self.peak_active, len(self.active))
            if current is None:
                continue
            family, position = current
            index = family.input_indices[position]
            self.active[device] = (family, position + 1)
            request = family.request_factory(index)
            if not isinstance(request, Mapping) or str(request.get("run_id")) != family.run_ids[position]:
                raise RuntimeError("SAM family factory changed its immutable original run identity")
            return device, index, family.family_id, request
        return None


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
    family_id: str | None = None


def execute_interpolation_tracker_task(
    context: object, kind: str, payload: Mapping[str, object],
) -> Mapping[str, object]:
    """Worker entry point: run raw observations and publish a packed artifact."""

    from .lta_experimental import run_mask_seed_session
    from .lta_outputs import write_json_atomically
    from .lta_rendering import LtaPhysicalViewCacheRef, render_native_tile_window
    from .lta_sam import SamInterpolationSessionPlan, prepare_sam_video_frames
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
    session = SamInterpolationSessionPlan(
        sequence_id=str(payload["run_id"]), session_index=0,
        frame_start=start, frame_stop=stop,
    )
    from .sam_resources import cpu_session_bytes
    image_side = int(getattr(getattr(context.predictor, "model", None), "image_size", 1008))
    cpu_admission = cpu_session_bytes(session.frame_count, shape[0] * shape[1], image_side=image_side)
    cpu_budget = _integer(payload.get("session_cpu_budget_bytes", 2 * 1024**3), "session_cpu_budget_bytes")
    if cpu_budget <= 0 or cpu_admission["estimated_peak_bytes"] > cpu_budget:
        raise MemoryError(f"SAM interpolation known CPU session buffers require {cpu_admission['estimated_peak_bytes']} "
                          f"bytes; admitted budget is {cpu_budget}; full interval was not started or truncated")
    cpu_admission["budget_bytes"] = cpu_budget
    render_started = time.perf_counter()
    resource = render_native_tile_window(
        cache_ref, frame_start=start, frame_stop=stop, tile_xyxy=crop,
    )
    render_seconds = time.perf_counter() - render_started
    image_cache_lifetime = dict(schema='xta.sam_image_cache_lifetime/1',
        gray_mapping_retired_after_render=getattr(resource, 'source_cache_mapping_retired', False) is True,
        model_input='independent_rgb_pil_frames',
        detach_allocation='existing_gray_to_rgb_repeat_counted_by_session_cpu_admission')
    input_started = time.perf_counter()
    input_preparation = prepare_sam_video_frames(context.predictor, resource,
        torch_module=getattr(context, 'torch_module', None),
        cuda_quota_bytes=getattr(context, 'sam_runtime', {}).get('cuda_allocator_quota', {}).get('limit_bytes'))
    input_seconds = time.perf_counter() - input_started
    if input_preparation['prepared']:
        image_cache_lifetime['model_input'] = input_preparation['clip_storage']
    loader_identity = (input_preparation['policy'], input_preparation.get('implementation_sha256'), image_side,
        tuple(input_preparation.get('image_mean', ())),
        tuple(input_preparation.get('image_std', ())))
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
            int(start) + int(local), loader_identity,
        ),
    )
    tracker_seconds = time.perf_counter() - tracker_started
    torch_mod = getattr(context, 'torch_module', None)
    cuda_quiescence = dict(synchronized=False, worker_local_device=0,
        scope='worker_process_cuda_context',
        execution_device_id=int(os.environ.get('LTA_EXECUTION_DEVICE_ID', '0')),
        worker_index=int(os.environ.get('LTA_WORKER_INDEX', '0')),
        run_id=str(payload['run_id']))
    if torch_mod is not None:
        torch_mod.cuda.synchronize(0)
        cuda_quiescence['synchronized'] = True
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
    sam_runtime = dict(_jsonable(context.sam_runtime))
    worker_startup = {key: sam_runtime.pop(key) for key in (
        'startup_cuda_quiescence', 'cuda_allocator_quota', 'resident_cpu_bytes') if key in sam_runtime}
    manifest = {
        "schema": "xta.sam-raw-tracker-run/1", "run_id": str(payload["run_id"]),
        "status": "invalid_removed_object" if bool(removed.any()) else "complete",
        "session_cpu_admission": cpu_admission,
        "prediction_valid": not bool(removed.any()),
        "coverage_complete": True, "crop_xyxy": list(crop), "seed_frame": prompt,
        "seed_artifact_sha256": str(payload["seed_sha256"]),
        "frame_range": [start, stop], "expected_frames": list(expected),
        "direction": direction, "image_cache": _image_cache_summary(cache_ref.payload()),
        "image_cache_lifetime": image_cache_lifetime,
        "input_preparation": input_preparation,
        "cuda_quiescence": cuda_quiescence,
        "raw_masks": {"path": str(mask_path), "sha256": _sha256(mask_path),
                      "size_bytes": mask_path.stat().st_size},
        "adapter_receipt": _jsonable(receipt),
        "profile": _jsonable(context.profile), "sam_runtime": sam_runtime,
        "worker_startup_receipt": worker_startup,
        "sam_model": _jsonable(getattr(context, "sam_model", {})),
        "request_metadata": _request_metadata(payload.get("request_metadata")),
        "feature_cache_before": _jsonable(cache_before),
        "feature_cache_after": _jsonable(None if feature_cache is None else feature_cache.snapshot()),
        "temporary_artifact_directory": str(output_dir),
        "timings": {
            "worker_started_monotonic": worker_started,
            "queue_seconds": max(0.0, worker_started - float(payload.get("submitted_monotonic", worker_started))),
            "render_seconds": render_seconds, "tracker_seconds": tracker_seconds,
            "input_preparation_seconds": input_seconds,
            "sdk_session_init_seconds": result["sdk_session_init_seconds"],
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


def _read_tracker_manifest(manifest_path: Path, *, expected_sha256: str | None = None):
    """Verify the small completion receipt independently of the raw transfer."""
    path = Path(manifest_path).resolve(strict=True)
    encoded = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise RuntimeError("SAM tracker manifest checksum changed in transfer")
    manifest = json.loads(encoded.decode("utf-8"))
    if manifest.get("schema") != "xta.sam-raw-tracker-run/1":
        raise RuntimeError("SAM tracker returned an unsupported evidence schema")
    return path, manifest


def _validate_completion_manifest(manifest, prepared, device, *, require_cuda, worker_index=None):
    """Bind a worker's quiescence proof to its exact admitted request."""
    if str(manifest.get("run_id")) != prepared.task.payload['run_id']:
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
    if any(manifest.get(key) != value for key, value in expected_contract.items()):
        raise RuntimeError("SAM raw evidence changed its immutable image/seed/geometry contract")
    if manifest.get("request_metadata", {}) != prepared.task.payload["request_metadata"]:
        raise RuntimeError("SAM raw evidence changed its immutable original-run/tile attribution")
    if worker_index is not None:
        actual_index = manifest.get('cuda_quiescence', {}).get('worker_index')
        if type(actual_index) is not int or actual_index != worker_index:
            raise RuntimeError('SAM completion changed its exact worker-slot context proof')
    if require_cuda:
        proof = manifest.get('cuda_quiescence', {})
        local_device = proof.get('worker_local_device')
        logical_device = proof.get('execution_device_id')
        if (proof.get('synchronized') is not True or type(local_device) is not int or local_device != 0
                or type(logical_device) is not int or logical_device != device
                or proof.get('run_id') != prepared.task.payload['run_id']):
            raise RuntimeError('SAM completion lacks exact CUDA-quiescence/device/run proof')


def load_tracker_run_result(
    manifest_path: Path, *, expected_sha256: str | None = None,
) -> SamTrackerRunResult:
    """Verify file-backed transfer and unpack complete attributable observations."""

    path, manifest = _read_tracker_manifest(manifest_path, expected_sha256=expected_sha256)
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


@dataclass(eq=False)
class _TrackerScope:
    token: str
    cache_ref: object
    cache_key: str
    capacity: int
    lookahead: int
    deferred: bool
    admission: object | None = None
    admission_limits: Mapping | None = None
    family_dispatch: _FamilyDispatch | None = None
    ready: deque = field(default_factory=deque)
    completed: deque = field(default_factory=deque)
    outstanding: dict = field(default_factory=dict)
    staging_directories: set = field(default_factory=set)
    submitted_order: list = field(default_factory=list)
    running: int = 0
    preparing: int = 0
    transfer_held: bool = False
    transfer_received_monotonic: float = 0.
    raw_decode_started_monotonic: float | None = None
    source_exhausted: bool = False
    closing: bool = False
    abandoned: bool = False
    consumed: int = 0

    @property
    def window(self):
        return self.capacity + self.lookahead

    @property
    def work_count(self):
        return len(self.outstanding) + self.preparing - int(self.transfer_held)


class _PoolScheduler:
    """One persistent pool API owner, fair scoped FIFO jobs and compact ACKs.

    Factories, profile validation, seed staging and raw decoding belong to the
    producing thread. This owner can spend only already admitted immutable
    envelopes. The global running inventory is bounded by physical workers;
    each scope separately bounds ready, running and unconsumed packets.
    """

    def __init__(self, tracker):
        self.tracker = tracker
        self._condition = tracker._state_condition
        self._pending = {}
        self._free = set(tracker.worker_slots)
        self._device_active = {device: 0 for device in tracker.device_ids}
        if tracker.workers_per_device == 2:
            tracker._capture_worker_inventory()
        self._round_robin = deque(tracker._scopes)
        self._error = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name='sam-pool-scheduler', daemon=True)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise self._error
        if self.tracker._cancel.is_set():
            raise RuntimeError(self.tracker._cancel_reason)
        if self._stopping:
            raise RuntimeError('SAM pool scheduler has stopped')

    def receive(self, scope):
        with self._condition:
            while True:
                self._check()
                if scope.completed:
                    if scope.transfer_held:
                        raise RuntimeError('SAM scope already owns a decoded transfer')
                    prepared, device, event = scope.completed.popleft()
                    scope.transfer_held = True
                    scope.transfer_received_monotonic = time.monotonic()
                    scope.raw_decode_started_monotonic = None
                    self._condition.notify_all()
                    return prepared, device, event
                self._condition.wait(timeout=0.05)

    def stop_and_join(self, timeout=1.0):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0., float(timeout)))
        if self._thread.is_alive():
            raise RuntimeError('SAM pool scheduler remains active; pool/GPU ownership retained')

    def _choose_scope(self):
        for _ in range(len(self._round_robin)):
            token = self._round_robin.popleft()
            scope = self.tracker._scopes.get(token)
            if scope is None:
                continue
            self._round_robin.append(token)
            if (scope.closing or not scope.ready or scope.running >= scope.capacity
                    or scope.deferred and (scope.completed or scope.transfer_held)):
                continue
            return scope
        return None

    def _dispatch(self, scope):
        from .lta_workers import LtaWorkerTask

        with self._condition:
            self._check()
            prepared = scope.ready[0]
            preferred = self.tracker._crop_affinity.get(prepared.affinity_key)
            candidates = sorted(self._free, key=lambda slot: (
                self._device_active[slot[0]], slot != preferred,
                self.tracker._slot_submissions[slot], slot))
        selected, lease = None, None
        for candidate in candidates:
            if self.tracker._compute_lease_factory is None:
                selected = candidate
                break
            device = candidate[0]
            with self.tracker._compute_lease_lock:
                existing = self.tracker._compute_leases.get(device)
            if existing is not None:
                selected = candidate
                break
            acquired = self.tracker._compute_lease_factory(device,
                f'SAM tracker compute {prepared.task.payload["run_id"]}')
            if acquired is not None:
                selected, lease = candidate, acquired
                break
        if selected is None:
            return False
        device, worker_index = selected
        identity = prepared.task.work_id, prepared.task.attempt_token
        if lease is not None:
            with self.tracker._compute_lease_lock:
                self.tracker._compute_leases[device] = lease
        # GPU credit is acquired only after CPU preparation has completed. From
        # this point uncertain submission keeps its fence until worker exit.
        with self._condition:
            self._check()
            if scope.closing or scope.ready[0] is not prepared:
                raise RuntimeError('SAM prepared scope changed before device dispatch')
            scope.ready.popleft()
            self._free.remove(selected)
            self._device_active[device] += 1
            scope.running += 1
            self._pending[identity] = scope, prepared, selected
        payload = dict(prepared.task.payload)
        payload['submitted_monotonic'] = time.monotonic()
        task = LtaWorkerTask(work_id=prepared.task.work_id,
            attempt_token=prepared.task.attempt_token, kind=prepared.task.kind, payload=payload)
        prepared = _SubmittedRun(prepared.input_index, task, prepared.output_directory,
            prepared.affinity_key, prepared.family_id)
        with self._condition:
            self._pending[identity] = scope, prepared, selected
            scope.outstanding[identity] = prepared
        self.tracker._pool.submit(task, execution_device_id=device,
            **({'worker_index': worker_index} if self.tracker.workers_per_device == 2 else {}))
        accepted_monotonic = time.monotonic()
        with self._condition:
            scope.submitted_order.append(prepared.input_index)
            self.tracker._device_submissions[device] += 1
            self.tracker._slot_submissions[selected] += 1
            if preferred == selected:
                self.tracker.dispatch_stats['affinity_hits'] += 1
            elif preferred is not None:
                self.tracker.dispatch_stats['affinity_lends'] += 1
            self.tracker._crop_affinity[prepared.affinity_key] = selected
            self.tracker._crop_affinity.move_to_end(prepared.affinity_key)
            if len(self.tracker._crop_affinity) > 4096:
                self.tracker._crop_affinity.popitem(last=False)
            self.tracker.dispatch_stats['submitted'] += 1
            if (scope.transfer_held and accepted_monotonic >= scope.transfer_received_monotonic
                    and (scope.raw_decode_started_monotonic is None
                         or accepted_monotonic <= scope.raw_decode_started_monotonic)):
                self.tracker.dispatch_stats['refilled_before_raw_transfer'] += 1
            self.tracker.dispatch_stats['peak_in_flight'] = max(
                self.tracker.dispatch_stats['peak_in_flight'], len(self._pending))
            if scope.family_dispatch is not None:
                scope.family_dispatch.execution_order.append(prepared.input_index)
                self.tracker.dispatch_stats['family_execution_order'] = list(scope.submitted_order)
            self._condition.notify_all()
        return True

    def _receive(self):
        event = self.tracker._pool.wait_result(timeout=0.05)
        identity = event.work_id, event.attempt_token
        with self._condition:
            owned = self._pending.get(identity)
            if owned is None:
                raise RuntimeError('SAM worker returned an unowned scoped task/attempt')
            scope, prepared, slot = owned
        device, worker_index = slot
        event_index = getattr(event, 'worker_index', 0)
        if (int(event.execution_device_id) != device or type(event_index) is not int
                or event_index != worker_index
                or self.tracker.workers_per_device == 2 and not hasattr(event, 'worker_index')):
            raise RuntimeError('SAM worker result changed its admitted device/worker-slot ownership')
        expected_pid = self.tracker._worker_pids.get(slot)
        if expected_pid is not None and (type(event.worker_pid) is not int or event.worker_pid != expected_pid):
            raise RuntimeError('SAM worker result changed its exact worker-process PID ownership')
        artifact_path = Path(event.artifact_path).resolve(strict=True)
        if artifact_path != prepared.output_directory.resolve(strict=True) / 'manifest.json':
            raise RuntimeError('SAM worker manifest escaped its assigned run staging directory')
        if (scope.admission_limits is not None and artifact_path.stat().st_size
                > int(scope.admission_limits['maximum_manifest_bytes'])):
            raise RuntimeError('SAM worker manifest exceeded its admitted compact packet bytes')
        validation_started = time.perf_counter()
        _path, manifest = _read_tracker_manifest(artifact_path, expected_sha256=event.artifact_sha256)
        _validate_completion_manifest(manifest, prepared, device,
            require_cuda=self.tracker._compute_lease_factory is not None,
            worker_index=worker_index if self.tracker.workers_per_device == 2 else None)
        packet = Path(str(manifest['raw_masks']['path'])).resolve(strict=True)
        if packet.parent != prepared.output_directory.resolve(strict=True):
            raise RuntimeError('SAM worker raw packet escaped its assigned scoped staging directory')
        if (scope.admission_limits is not None and packet.stat().st_size
                > int(scope.admission_limits['maximum_packed_packet_bytes'])):
            raise RuntimeError('SAM worker raw packet exceeded its admitted packed-mask bytes')
        gray_retired = manifest.get('image_cache_lifetime', {}).get(
            'gray_mapping_retired_after_render') is True
        del manifest
        if self._device_active[device] <= 0:
            raise RuntimeError('SAM physical worker-slot activity count underflow')
        # synchronize(0) proves this process context only. A physical lease is
        # returned after the last exact context ACK, never while a peer remains.
        if self.tracker._compute_lease_factory is not None and self._device_active[device] == 1:
            with self.tracker._compute_lease_lock:
                lease = self.tracker._compute_leases.pop(device)
            try:
                self.tracker._compute_lease_release(lease)
            except BaseException:
                with self.tracker._compute_lease_lock:
                    self.tracker._compute_leases[device] = lease
                raise
        with self._condition:
            del self._pending[identity]
            self._device_active[device] -= 1
            scope.running -= 1
            self._free.add(slot)
            if identity not in scope.outstanding or len(scope.completed) >= scope.window:
                raise RuntimeError('SAM scoped completion exceeded its charged packet window')
            scope.completed.append((prepared, slot, event))
            proof = self.tracker._source_cache_retirement_proofs[scope.cache_key]
            proof['all_gray_mappings_retired'] &= gray_retired
            proof['completed_runs'] += 1
            self.tracker.dispatch_stats['completion_manifest_validation_seconds'] += (
                time.perf_counter()-validation_started)
            self.tracker.dispatch_stats['completion_acks_pumped'] += 1
            self._condition.notify_all()

    def _run(self):
        try:
            while True:
                with self._condition:
                    if self._stopping or self.tracker._cancel.is_set():
                        return
                    scope = self._choose_scope() if self._free else None
                    pending = bool(self._pending)
                if scope is not None and self._dispatch(scope):
                    continue
                if pending:
                    try:
                        self._receive()
                    except TimeoutError:
                        pass
                else:
                    with self._condition:
                        self._condition.wait(timeout=0.05)
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
            if not self.tracker._cancel.is_set():
                try:
                    self.tracker.cancel(f'SAM pool scheduler failed: {error}')
                except BaseException as cleanup_error:
                    if callable(getattr(error, 'add_note', None)):
                        error.add_note(f'SAM scheduler quarantine also failed: {cleanup_error}')
        finally:
            with self._condition:
                self._condition.notify_all()


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
        workers_per_device: int = 1,
        cuda_allocator_fractions: Mapping[str, float] | None = None,
        feature_cache_bytes: int = 512 * 1024 * 1024,
        feature_cache_headroom_bytes: int | None = None,
        session_cpu_budget_bytes: int = 2 * 1024**3,
        compute_lease_factory=None, compute_lease_release=None,
        residency_quarantine=None, before_worker_shutdown=None, after_worker_shutdown=None,
    ) -> None:
        self.model_path = str(model_path)
        self.device_ids = tuple(_integer(value, "device_id") for value in device_ids)
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids) or min(self.device_ids) < 0:
            raise ValueError("SAM tracker requires unique nonnegative CUDA devices")
        self.workers_per_device = _integer(workers_per_device, 'workers_per_device')
        if self.workers_per_device not in (1, 2):
            raise ValueError('SAM workers_per_device must be one or two isolated predictor processes')
        self.worker_slots = tuple((device, index) for device in self.device_ids
            for index in range(self.workers_per_device))
        self.worker_count = len(self.worker_slots)
        self.cuda_allocator_fractions = None
        if cuda_allocator_fractions is not None:
            if self.workers_per_device != 2 or not isinstance(cuda_allocator_fractions, Mapping):
                raise ValueError('SAM allocator fractions require dual predictor processes')
            fractions = dict(cuda_allocator_fractions)
            if set(fractions) != {str(device) for device in self.device_ids} or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= .5 for value in fractions.values()):
                raise ValueError('SAM dual allocator fractions must name every device and be finite in (0, 0.5]')
            self.cuda_allocator_fractions = fractions
        self.artifact_root = Path(artifact_root).resolve(strict=False)
        self.profile = str(profile)
        self.startup_timeout = float(startup_timeout)
        self.session_cpu_budget_bytes = _integer(session_cpu_budget_bytes, "session_cpu_budget_bytes")
        if self.session_cpu_budget_bytes <= 0:
            raise ValueError("SAM session CPU byte budget must be positive")
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
        self._state_condition = threading.Condition(threading.RLock())
        self._scopes = {}
        self._pool_lifecycle_lock = threading.RLock()
        self._compute_lease_lock = threading.RLock()
        self._scheduler = None
        self._crop_affinity: OrderedDict[tuple[object, ...], int] = OrderedDict()
        self._device_submissions = {device: 0 for device in self.device_ids}
        self._slot_submissions = {slot: 0 for slot in self.worker_slots}
        self._worker_pids = {}
        self._startup_receipts = ()
        self.dispatch_stats = {
            "submitted": 0, "completed": 0, "peak_in_flight": 0,
            "affinity_hits": 0, "affinity_lends": 0, "wait_seconds": 0.0,
            "completion_manifest_validation_seconds": 0.0,
            "raw_transfer_decode_seconds": 0.0, "refilled_before_raw_transfer": 0,
            "completion_acks_pumped": 0,
        }
        self._source_cache_ref = None
        self._source_cache_retirement_proofs = {}
        self._compute_lease_factory = compute_lease_factory
        self._compute_lease_release = compute_lease_release
        self._residency_quarantine = residency_quarantine
        self._before_worker_shutdown = before_worker_shutdown
        self._after_worker_shutdown = after_worker_shutdown
        self._compute_leases = {}
        self.startup_cuda_quiescent = False
        if source_cache_ref is not None:
            self.set_source_cache(source_cache_ref)
        _register_sam_scheduler_sample(self)

    def snapshot(self):
        """Compact ownership counts; running includes worker CPU/packing phases."""
        with self._state_condition:
            scopes = tuple(self._scopes.values())
            return dict(sample_monotonic_ns=time.monotonic_ns(),
                active_scopes=len(scopes), credited_scopes=sum(scope.admission is not None for scope in scopes),
                ready_jobs=sum(len(scope.ready) for scope in scopes),
                preparing_jobs=sum(scope.preparing for scope in scopes),
                running_jobs=sum(scope.running for scope in scopes),
                acked_awaiting_consumer_jobs=sum(len(scope.completed) for scope in scopes),
                consumer_held_jobs=sum(scope.transfer_held for scope in scopes),
                live_prepared_bank_bytes=sum(int(scope.admission_limits['prepared_bank_bytes'])
                    for scope in scopes if scope.admission_limits is not None),
                submitted_jobs=int(self.dispatch_stats['submitted']),
                completed_jobs=int(self.dispatch_stats['completed']),
                completion_acks=int(self.dispatch_stats['completion_acks_pumped']),
                closed=self._closed, cancelled=self._cancel.is_set())

    def set_source_cache(self, cache_ref: object) -> None:
        from .lta_rendering import LtaPhysicalViewCacheRef

        if not isinstance(cache_ref, LtaPhysicalViewCacheRef):
            raise TypeError("SAM source_cache_ref must be a LtaPhysicalViewCacheRef")
        cache_ref.revalidate()
        with self._state_condition:
            if self._scopes:
                raise RuntimeError("cannot replace SAM source cache while a bounded iterator is active")
            self._source_cache_ref = cache_ref

    @property
    def _iteration_active(self):
        with self._state_condition:
            return bool(self._scopes)

    def release_source_cache(self, cache_ref: object) -> dict[str, object]:
        """Prove complete iterator consumption before a cache owner unlinks it."""
        from .lta_rendering import LtaPhysicalViewCacheRef
        if not isinstance(cache_ref, LtaPhysicalViewCacheRef):
            raise TypeError('SAM cache retirement requires an immutable cache descriptor')
        key = hashlib.sha256(json.dumps(cache_ref.payload(), sort_keys=True).encode()).hexdigest()
        with self._state_condition:
            proof = self._source_cache_retirement_proofs.get(key)
            if proof is not None and proof.get('active_scopes', 0):
                raise RuntimeError('SAM image cache still has an active scoped iterator owner')
            proof = self._source_cache_retirement_proofs.pop(key, None)
            if proof is None:
                proof = dict(complete=True, all_gray_mappings_retired=True, completed_runs=0)
                basis = 'no_submitted_consumers'
            elif self._closed and self.workers_settled:
                proof = dict(proof, complete=True, all_gray_mappings_retired=True)
                basis = 'worker_processes_exited'
            else:
                basis = 'completed_iterator_and_detached_worker_rgb'
            if not proof['complete'] or not proof['all_gray_mappings_retired']:
                self._source_cache_retirement_proofs[key] = proof
                raise RuntimeError('SAM image cache has no completed worker mapping-retirement proof')
            if self._source_cache_ref == cache_ref:
                self._source_cache_ref = None
            return dict(schema='xta.sam_image_cache_retirement/1', status='retired',
                completed_runs=int(proof['completed_runs']), workers_finished=True,
                gray_mappings_retired=True, model_and_feature_cache_retained=not self._closed,
                proof_basis=basis)

    def start(self) -> "SamInterpolationTracker":
        with self._pool_lifecycle_lock:
            return self._start_owned()

    def _capture_worker_inventory(self):
        ready = tuple(self._pool.ready_events)
        slots = tuple((getattr(event, 'execution_device_id', None), getattr(event, 'worker_index', None)) for event in ready)
        if (len(slots) != self.worker_count or set(slots) != set(self.worker_slots)
                or any(type(device) is not int or type(index) is not int for device, index in slots)
                or any(type(getattr(event, 'worker_pid', None)) is not int
                    or event.worker_pid <= 0 for event in ready)
                or len({event.worker_pid for event in ready}) != self.worker_count):
            raise RuntimeError('SAM startup has no exact worker-slot/process readiness inventory')
        self._worker_pids = {slot: event.worker_pid for slot, event in zip(slots, ready)}
        self._startup_receipts = tuple(dict(execution_device_id=event.execution_device_id,
            worker_index=event.worker_index, worker_pid=event.worker_pid,
            sam_runtime=copy.deepcopy(event.metadata.get('sam_runtime', {}))) for event in ready)
        if self.cuda_allocator_fractions is not None:
            for event in ready:
                quota = event.metadata.get('sam_runtime', {}).get('cuda_allocator_quota', {})
                fraction = self.cuda_allocator_fractions[str(event.execution_device_id)]
                total, limit = quota.get('cuda_total_bytes'), quota.get('limit_bytes')
                if (quota.get('schema') != 'xta.sam_cuda_allocator_quota/1'
                        or quota.get('enforcement') != 'torch_caching_allocator_fraction'
                        or type(quota.get('execution_device_id')) is not int
                        or quota['execution_device_id'] != event.execution_device_id
                        or type(quota.get('worker_index')) is not int or quota['worker_index'] != event.worker_index
                        or quota.get('fraction') != fraction or type(total) is not int or total <= 0
                        or type(limit) is not int or limit != int(total*fraction) or limit <= 0
                        or any(type(quota.get(key)) is not int or not 0 <= quota[key] <= limit
                            for key in ('allocated_bytes', 'reserved_bytes'))
                        or type(quota.get('cuda_free_bytes')) is not int
                        or not 0 <= quota['cuda_free_bytes'] <= total):
                    raise RuntimeError('SAM dual startup has no exact enforced allocator-quota/slot proof')
        if self._compute_lease_factory is not None:
            for event in ready:
                proof = event.metadata.get('sam_runtime', {}).get('startup_cuda_quiescence', {})
                if (proof.get('synchronized') is not True
                        or type(proof.get('worker_local_device')) is not int or proof['worker_local_device'] != 0
                        or self.workers_per_device == 2 and (
                            type(proof.get('worker_index')) is not int or proof['worker_index'] != event.worker_index)):
                    raise RuntimeError('SAM startup has no exact worker CUDA-quiescence/slot proof')
            self.startup_cuda_quiescent = True

    @property
    def startup_receipts(self):
        return copy.deepcopy(self._startup_receipts)

    def _start_owned(self):
        from .lta_workers import LtaWorkerInit, LtaWorkerPool

        if self._closed:
            raise RuntimeError("SAM tracker has been closed")
        if self._cancel.is_set():
            raise RuntimeError(self._cancel_reason)
        if self._pool is None:
            if self.workers_per_device == 2 and self.cuda_allocator_fractions is None:
                raise RuntimeError('Dual SAM startup requires measured per-process CUDA allocator fractions')
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
                                        "feature_cache_headroom_bytes": self.feature_cache_headroom_bytes,
                                        **({"cuda_allocator_fractions": self.cuda_allocator_fractions}
                                            if self.cuda_allocator_fractions is not None else {})},
                    ), startup_timeout=self.startup_timeout, cancel_event=self._cancel,
                    workers_per_device=self.workers_per_device,
                )
            except BaseException as error:
                self._pool = getattr(error, "unsettled_worker_pool", None)
                self._residency_released = self._pool is None
                raise
            self._capture_worker_inventory()
        if self._scheduler is None:
            with self._state_condition:
                self._scheduler = _PoolScheduler(self)
        return self

    @property
    def workers_settled(self) -> bool:
        return self._residency_released if self._pool is None else bool(self._pool.workers_settled)

    @property
    def residency_released(self) -> bool:
        return self.workers_settled

    def _settle_pool(self, *, timeout: float) -> None:
        with self._pool_lifecycle_lock:
            self._stop_scheduler()
            self._settle_pool_owned(timeout=timeout)
            self._retire_abandoned_scopes()

    def _stop_scheduler(self) -> None:
        """Pool shutdown may start only after its API owner has joined."""
        scheduler = self._scheduler
        if scheduler is not None:
            scheduler.stop_and_join(timeout=1.0)
            if self._scheduler is scheduler:
                self._scheduler = None

    def _settle_pool_owned(self, *, timeout: float) -> None:
        pool = self._pool
        if pool is None:
            return
        if self._before_worker_shutdown is not None:
            self._before_worker_shutdown()
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
        if self._residency_released and self._after_worker_shutdown is not None:
            self._after_worker_shutdown()
        if self._residency_released:
            self._pool = None
            if self._compute_lease_release is not None:
                with self._compute_lease_lock:
                    leases = tuple(self._compute_leases.values())
                    self._compute_leases.clear()
                for lease in leases:
                    self._compute_lease_release(lease)
        if original_error is not None:
            raise original_error
        if not self._residency_released:
            raise RuntimeError("SAM model residency is unproven after worker shutdown")

    def _prepare_task(
        self, request: Mapping[str, object], *, cache_ref: object,
        input_index: int, staging_directories: set[Path],
    ) -> _SubmittedRun:
        from .lta_sam import SamInterpolationSessionPlan
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
        SamInterpolationSessionPlan(sequence_id=str(run_id), session_index=0, frame_start=start, frame_stop=stop)
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
        from .sam_resources import cpu_session_bytes
        cpu_budget = self.session_cpu_budget_bytes
        resource_profile = request.get("resource_profile")
        if resource_profile is not None:
            from .sam_resources import validate_live_sam_resource_profile
            assigned = validate_live_sam_resource_profile(resource_profile)
            cpu_budget = int(assigned["assigned_session_cpu_bytes"])
        cpu_admission = cpu_session_bytes(stop - start, (y1 - y0) * (x1 - x0))
        if cpu_admission["estimated_peak_bytes"] > cpu_budget:
            raise MemoryError(f"SAM interpolation known CPU session buffers require {cpu_admission['estimated_peak_bytes']} "
                              f"bytes; admitted budget is {cpu_budget}; full interval was not staged or truncated")
        metadata = _request_metadata(request.get("metadata"))
        token = uuid.uuid4().hex
        with self._state_condition:
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
                     "session_cpu_budget_bytes": cpu_budget,
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

    def _register_scope(self, cache_ref, *, capacity, deferred, admission, family_dispatch):
        """Claim bounded producer ownership before invoking any request factory."""
        if admission is None:
            capacity = min(capacity, len(self.device_ids))
        acquired = False
        try:
            lookahead = 0
            if admission is not None:
                from .sam_resources import validate_sam_tracker_scope_admission
                limits = validate_sam_tracker_scope_admission(admission)
                admission.validate_cache(cache_ref)
                admission.acquire_scope()
                acquired = True
                if capacity > int(limits['max_in_flight']):
                    raise RuntimeError('SAM scope exceeds its live CPU wave admission')
                if deferred != bool(limits['defer_refill_until_consumed']):
                    raise RuntimeError('SAM scope changed its admitted raw-transfer barrier')
                lookahead = _integer(limits['lookahead_jobs'], 'lookahead_jobs')
                if not 0 <= lookahead <= capacity:
                    raise RuntimeError('SAM scope has an invalid prepared-input bank')
            key = hashlib.sha256(json.dumps(cache_ref.payload(), sort_keys=True).encode()).hexdigest()
            with self._state_condition:
                # Uncredited legacy callers own only the historical single raw
                # consumer margin. They cannot overlap another scope's model or
                # transfer allocation; credited scopes all remain concurrent.
                while self._scopes and (admission is None or any(
                        scope.admission is None for scope in self._scopes.values())):
                    if self._closed or self._cancel.is_set():
                        raise RuntimeError(self._cancel_reason if self._cancel.is_set()
                            else 'SAM tracker has been closed')
                    self._state_condition.wait(timeout=0.05)
                if self._closed or self._cancel.is_set():
                    raise RuntimeError(self._cancel_reason if self._cancel.is_set()
                        else 'SAM tracker has been closed')
                scope = _TrackerScope(uuid.uuid4().hex, cache_ref, key, capacity,
                    lookahead, deferred, admission, limits if admission is not None else None,
                    family_dispatch)
                proof = self._source_cache_retirement_proofs.setdefault(key, dict(
                    complete=False, all_gray_mappings_retired=True, completed_runs=0,
                    active_scopes=0, failed_scopes=0))
                proof['active_scopes'] += 1
                proof['complete'] = False
                self._scopes[scope.token] = scope
                if self._scheduler is not None:
                    self._scheduler._round_robin.append(scope.token)
                self._state_condition.notify_all()
                return scope
        except BaseException:
            if acquired:
                admission.release_scope()
            raise

    def _unregister_scope(self, scope, *, complete):
        """Retire only settled source/packet owners, releasing credit afterward."""
        with self._state_condition:
            if self._scopes.get(scope.token) is not scope:
                raise RuntimeError('SAM scoped iterator lost its ownership registration')
            proof = self._source_cache_retirement_proofs[scope.cache_key]
            if proof['active_scopes'] <= 0:
                raise RuntimeError('SAM scoped image owner count underflow')
        if scope.admission is not None:
            scope.admission.release_scope()
        with self._state_condition:
            if self._scopes.pop(scope.token, None) is not scope:
                raise RuntimeError('SAM scoped iterator changed during credit retirement')
            if self._scheduler is not None:
                try:
                    self._scheduler._round_robin.remove(scope.token)
                except ValueError:
                    pass
            proof['active_scopes'] -= 1
            proof['failed_scopes'] += int(not complete)
            proof['complete'] = proof['active_scopes'] == 0 and proof['failed_scopes'] == 0
            scope.outstanding.clear()
            scope.ready.clear()
            scope.completed.clear()
            self._state_condition.notify_all()

    def _retire_abandoned_scopes(self):
        if not self.workers_settled:
            return
        with self._state_condition:
            abandoned = tuple(scope for scope in self._scopes.values()
                if scope.abandoned and not scope.preparing)
        for scope in abandoned:
            for directory in tuple(scope.staging_directories):
                self._remove_staging(directory)
            scope.staging_directories.clear()
            self._unregister_scope(scope, complete=False)

    def iter_results(
        self, requests: Iterable[Mapping[str, object]], *,
        source_cache_ref: object | None = None, max_in_flight: int | None = None,
        defer_refill_until_consumed: bool = False,
        scope_admission: object | None = None,
        _family_dispatch: _FamilyDispatch | None = None,
    ) -> Iterator[tuple[int, SamTrackerRunResult]]:
        """Yield exact scoped completions through the persistent fair scheduler.

        Live producer admission funds each scope's CPU wave, one transfer and
        any additional immutable prepared bank. Preparation/factories/decoding
        stay on this original thread. While its CPU consumer is paused, the
        pool owner may dispatch its charged ready bank or another ready scope.
        An uncredited legacy iterator remains globally exclusive, preserving
        its historical N+1 packet/one transfer ownership rather than inventing
        additional CPU credit. All paths share the same scheduler and proofs.
        """
        from .lta_rendering import LtaPhysicalViewCacheRef
        from .lta_workers import LtaWorkerTask

        capacity = self.worker_count if max_in_flight is None else _integer(max_in_flight, 'max_in_flight')
        if not isinstance(defer_refill_until_consumed, bool):
            raise TypeError('SAM refill deferral must be boolean')
        if not 1 <= capacity <= self.worker_count:
            raise ValueError('max_in_flight must be between one and the admitted worker count')
        with self._state_condition:
            cache_ref = source_cache_ref if source_cache_ref is not None else self._source_cache_ref
        if not isinstance(cache_ref, LtaPhysicalViewCacheRef):
            raise RuntimeError('SAM tracker requires an immutable image cache for this iterator')
        cache_ref.revalidate()
        scope = self._register_scope(cache_ref, capacity=capacity,
            deferred=defer_refill_until_consumed, admission=scope_admission,
            family_dispatch=_family_dispatch)
        capacity = scope.capacity
        complete = False
        next_index = 0
        family_lane = 0
        iterator = None
        seen_run_ids = set()

        def prepare_bank():
            nonlocal next_index, family_lane
            while True:
                with self._state_condition:
                    if self._closed or self._cancel.is_set():
                        raise RuntimeError(self._cancel_reason if self._cancel.is_set()
                            else 'SAM tracker has been closed')
                    if scope.source_exhausted or scope.work_count >= scope.window:
                        return
                    # This reservation precedes the lazy factory and every
                    # seed/cached-source/NPZ allocation. Its credit was already
                    # acquired on the producing thread at scope registration.
                    scope.preparing += 1
                prepared = None
                reserved = True
                try:
                    family_id = None
                    if _family_dispatch is None:
                        try:
                            request = next(iterator)
                        except StopIteration:
                            with self._state_condition:
                                scope.source_exhausted = True
                            return
                        input_index = next_index
                    else:
                        chosen = None
                        for _ in range(capacity):
                            lane = family_lane % capacity
                            family_lane += 1
                            chosen = _family_dispatch.next_for((lane,))
                            if chosen is not None:
                                break
                        if chosen is None:
                            with self._state_condition:
                                scope.source_exhausted = True
                            return
                        _lane, input_index, family_id, request = chosen
                    if scope.admission is not None:
                        crop = tuple(request['crop_xyxy'])
                        scope.admission.validate_request(
                            _integer(request['frame_stop'], 'frame_stop')-_integer(request['frame_start'], 'frame_start'),
                            (crop[2]-crop[0])*(crop[3]-crop[1]))
                    prepared = self._prepare_task(request, cache_ref=cache_ref,
                        input_index=input_index, staging_directories=scope.staging_directories)
                    del request
                    if (scope.admission_limits is not None
                            and Path(prepared.task.payload['seed_path']).stat().st_size
                                > int(scope.admission_limits['maximum_seed_npz_bytes'])):
                        raise RuntimeError('SAM seed staging exceeded its admitted immutable packet bytes')
                    scientific_id = prepared.task.payload['run_id']
                    if scientific_id in seen_run_ids:
                        raise ValueError(f'duplicate SAM run_id in one iterator: {scientific_id}')
                    seen_run_ids.add(scientific_id)
                    transport_id = f'sam-scope-{scope.token}:{scientific_id}'
                    payload = dict(prepared.task.payload)
                    if scope.admission_limits is not None:
                        payload['session_cpu_budget_bytes'] = min(
                            payload['session_cpu_budget_bytes'],
                            int(scope.admission_limits['maximum_session_cpu_bytes']))
                    task = LtaWorkerTask(work_id=transport_id, attempt_token=prepared.task.attempt_token,
                        kind=prepared.task.kind, payload=payload)
                    prepared = _SubmittedRun(prepared.input_index, task,
                        prepared.output_directory, prepared.affinity_key, family_id)
                    with self._pool_lifecycle_lock:
                        self.start()
                        if self._closed or self._cancel.is_set():
                            raise RuntimeError(self._cancel_reason if self._cancel.is_set()
                                else 'SAM tracker has been closed')
                        if self._pool is None:
                            raise RuntimeError('SAM startup did not provide an admitted worker pool')
                        if self._scheduler is None:
                            with self._state_condition:
                                self._scheduler = _PoolScheduler(self)
                    with self._state_condition:
                        self._scheduler._check()
                        identity = task.work_id, task.attempt_token
                        if identity in scope.outstanding:
                            raise RuntimeError('SAM scope duplicated its prepared transport attempt')
                        scope.preparing -= 1
                        reserved = False
                        scope.outstanding[identity] = prepared
                        scope.ready.append(prepared)
                        next_index += 1
                        if _family_dispatch is not None:
                            self.dispatch_stats['family_assigned'] = _family_dispatch.assigned_families
                            self.dispatch_stats['family_active_peak'] = _family_dispatch.peak_active
                        self._state_condition.notify_all()
                finally:
                    if reserved:
                        with self._state_condition:
                            scope.preparing -= 1
                            self._state_condition.notify_all()

        try:
            iterator = iter(requests)
            prepare_bank()
            while scope.outstanding or not scope.source_exhausted:
                with self._state_condition:
                    if self._cancel.is_set():
                        raise RuntimeError(self._cancel_reason)
                prepare_bank()
                if not scope.outstanding:
                    continue
                started = time.perf_counter()
                prepared, slot, event = self._scheduler.receive(scope)
                device, worker_index = slot
                with self._state_condition:
                    self.dispatch_stats['wait_seconds'] += time.perf_counter()-started
                if not defer_refill_until_consumed:
                    prepare_bank()
                artifact_path = Path(event.artifact_path).resolve(strict=True)
                started = time.perf_counter()
                with self._state_condition:
                    scope.raw_decode_started_monotonic = time.monotonic()
                result = load_tracker_run_result(artifact_path, expected_sha256=event.artifact_sha256)
                _validate_completion_manifest(result.receipt, prepared, device,
                    require_cuda=self._compute_lease_factory is not None,
                    worker_index=worker_index if self.workers_per_device == 2 else None)
                receipt = dict(result.receipt)
                receipt['dispatch'] = dict(input_index=prepared.input_index,
                    execution_device_id=device, worker_pid=event.worker_pid,
                    worker_index=worker_index,
                    scope_token=scope.token, refill_after_consumption=defer_refill_until_consumed,
                    prepared_lookahead_jobs=scope.lookahead)
                if prepared.family_id is not None:
                    receipt['dispatch']['family_id'] = prepared.family_id
                result = SamTrackerRunResult(result.frames, result.tracker_scores,
                    result.observation_status, receipt)
                with self._state_condition:
                    self.dispatch_stats['raw_transfer_decode_seconds'] += time.perf_counter()-started
                    self.dispatch_stats['completed'] += 1
                yield prepared.input_index, result
                self._remove_staging(prepared.output_directory)
                scope.staging_directories.discard(prepared.output_directory)
                del result
                with self._state_condition:
                    identity = prepared.task.work_id, prepared.task.attempt_token
                    if scope.outstanding.pop(identity, None) is None or not scope.transfer_held:
                        raise RuntimeError('SAM scope lost its consumed raw-packet owner')
                    scope.transfer_held = False
                    scope.consumed += 1
                    self._state_condition.notify_all()
            if self._cancel.is_set():
                raise RuntimeError(self._cancel_reason)
            if len(scope.submitted_order) != next_index or scope.consumed != next_index:
                raise RuntimeError('SAM scoped submission/consumption inventory did not settle')
            if _family_dispatch is not None:
                if next_index != _family_dispatch.total_jobs:
                    raise RuntimeError('SAM family dispatcher omitted immutable original jobs')
                _family_dispatch.retire_completed(range(capacity))
                if (_family_dispatch.active or _family_dispatch.completed_families
                        != _family_dispatch.assigned_families):
                    raise RuntimeError('SAM family completion inventory did not settle after jobs drained')
                with self._state_condition:
                    self.dispatch_stats['family_completed'] = _family_dispatch.completed_families
            complete = True
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            with self._state_condition:
                scope.closing = True
                self._state_condition.notify_all()
            if not complete:
                try:
                    self.cancel('SAM scoped generation failed or its consumer stopped before completion')
                    self._settle_pool(timeout=1.0)
                except BaseException as error:
                    cleanup_error = error
                self._closed = True
            settled = complete or self.workers_settled and self._scheduler is None
            if settled:
                try:
                    for directory in tuple(scope.staging_directories):
                        self._remove_staging(directory)
                    scope.staging_directories.clear()
                    self._unregister_scope(scope, complete=complete)
                    if self.workers_settled and self._compute_lease_release is not None:
                        with self._compute_lease_lock:
                            leases = tuple(self._compute_leases.values())
                            self._compute_leases.clear()
                        for lease in leases:
                            self._compute_lease_release(lease)
                except BaseException as error:
                    cleanup_error = cleanup_error or error
                    with self._state_condition:
                        if self._scopes.get(scope.token) is scope:
                            scope.abandoned = True
                            self._state_condition.notify_all()
            else:
                with self._state_condition:
                    scope.abandoned = True
                    self._state_condition.notify_all()
            if cleanup_error is not None:
                if primary_error is None or isinstance(primary_error, GeneratorExit):
                    raise cleanup_error
                if callable(getattr(primary_error, 'add_note', None)):
                    primary_error.add_note(
                        f'SAM bounded scope cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}')

    def iter_family_results(
        self, families: Sequence[SamTrackerFamily], *,
        source_cache_ref: object | None = None, max_in_flight: int | None = None,
        defer_refill_until_consumed: bool = False, schedule: str = "fifo",
        scope_admission: object | None = None,
        execution_order_callback: Callable[[tuple[int, ...]], object] | None = None,
    ) -> Iterator[tuple[int, SamTrackerRunResult]]:
        """Family-local dispatch, yielding immutable original indices.

        Virtual producer lanes preserve sealed family/index identities. Actual
        devices are chosen by the shared scheduler; cache affinity is a soft
        preference and cannot block otherwise ready work on an idle worker.
        LPT uses declared frame work only and is a diagnostic ordering option.
        Flat/tiled callers continue to use the existing ``iter_results`` path.
        The optional callback receives this iterator's immutable submission
        order after all results are consumed successfully. Scope provenance
        must use this receipt rather than the shared diagnostic statistics.
        """
        if not isinstance(families, (tuple, list)):
            raise TypeError("SAM family inventory must be a finite metadata sequence")
        if execution_order_callback is not None and not callable(execution_order_callback):
            raise TypeError("SAM execution-order callback must be callable")
        dispatcher = _FamilyDispatch(families, schedule)
        capacity = self.worker_count if max_in_flight is None else max_in_flight
        yield from self.iter_results((), source_cache_ref=source_cache_ref,
            max_in_flight=capacity, defer_refill_until_consumed=defer_refill_until_consumed,
            scope_admission=scope_admission,
            _family_dispatch=dispatcher)
        if execution_order_callback is not None:
            execution_order_callback(tuple(dispatcher.execution_order))

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
        with self._state_condition:
            active = bool(self._scopes)
            was_cancelled = self._cancel.is_set()
            self._closed = True
            self._state_condition.notify_all()
        error = None
        if not self._cancel.is_set():
            try:
                self.cancel('SAM tracker closed during bounded generation' if active
                    else 'SAM tracker has been closed')
            except BaseException as caught:
                error = caught
        try:
            self._settle_pool(timeout=1.0 if active or was_cancelled else 30.0)
        except BaseException as caught:
            if error is None:
                error = caught
            elif callable(getattr(error, 'add_note', None)):
                error.add_note(f'SAM worker shutdown also failed: {type(caught).__name__}: {caught}')
        if error is not None:
            raise error

    def cancel(self, reason: str = "SAM tracker operation cancelled") -> None:
        """Interrupt parent-side startup/result waits without a queue race.

        The inference-owning thread exits with an infrastructure error. Its
        owner then closes the pool after joining consumers; partial artifacts
        are diagnostic data and never a completed empty support layer.
        """

        self._cancel_reason = str(reason)
        self._cancel.set()
        with self._state_condition:
            self._state_condition.notify_all()
        if self._residency_quarantine is not None:
            self._residency_quarantine(str(reason))

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

    quota_bytes = None
    fraction = None
    if config.get('cuda_allocator_fractions') is not None:
        import torch
        fractions = config['cuda_allocator_fractions']
        fraction = fractions.get(os.environ.get('LTA_EXECUTION_DEVICE_ID', '0'))
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 < fraction <= .5):
            raise ValueError('SAM worker has no valid dual-process CUDA allocator fraction')
        torch.cuda.set_per_process_memory_fraction(float(fraction), 0)
        quota_bytes = int(torch.cuda.mem_get_info(0)[1] * fraction)
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
                available = int(free_bytes) + reusable
                if quota_bytes is not None:
                    available = min(available, max(0, quota_bytes-int(torch.cuda.memory_allocated(0))))
                return available

            budget = min(requested_cache_bytes, max(0, memory_probe() - headroom))
            context.feature_cache = LruTrackerFeatureCache(
                max_bytes=budget, memory_probe=memory_probe, headroom_bytes=headroom,
                model_identity=json.dumps({"model": context.sam_model, "runtime": context.sam_runtime,
                                           "profile": context.profile}, sort_keys=True),
                position_source_identity=str(context.sam_runtime.get("package_tree_sha256", "")),
            )
        else:
            context.feature_cache = None
        context.torch_module.cuda.synchronize(0)
        context.sam_runtime = dict(context.sam_runtime, startup_cuda_quiescence={
            'synchronized': True, 'worker_local_device': 0,
            'scope': 'worker_process_cuda_context',
            'worker_index': int(os.environ.get('LTA_WORKER_INDEX', '0'))})
        if quota_bytes is not None:
            torch = context.torch_module
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            context.sam_runtime['cuda_allocator_quota'] = dict(schema='xta.sam_cuda_allocator_quota/1',
                fraction=fraction, limit_bytes=quota_bytes, cuda_total_bytes=int(total_bytes),
                cuda_free_bytes=int(free_bytes), allocated_bytes=int(torch.cuda.memory_allocated(0)),
                reserved_bytes=int(torch.cuda.memory_reserved(0)),
                worker_index=int(os.environ.get('LTA_WORKER_INDEX', '0')),
                execution_device_id=int(os.environ.get('LTA_EXECUTION_DEVICE_ID', '0')),
                enforcement='torch_caching_allocator_fraction')
        try:
            import psutil
            context.sam_runtime['resident_cpu_bytes'] = int(psutil.Process().memory_info().rss)
        except (ImportError, OSError):
            context.sam_runtime['resident_cpu_bytes'] = None
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
