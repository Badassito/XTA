"""Observation-only SAM bridge generation and selected directional publication.

This module deliberately imports neither SAM nor the SDF generator at import
time. A tracker generates complete raw proposals; the proposal policy selects
contributors before any additive volume or parent-gate support is published.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Any

import numpy as np


class SamInterpolationInfrastructureError(RuntimeError):
    """A generation failure, which must never become an empty successful bridge."""


def _check_cancelled(cancel_event: object) -> None:
    if cancel_event is not None and bool(cancel_event.is_set()):
        raise SamInterpolationInfrastructureError("SAM interpolation cancelled before complete publication")


def _failure_receipt(destination: Path, error: BaseException, phase: str, generated_runs: int) -> None:
    failure = {"status": "infrastructure_invalid", "complete": False,
               "backend": "sam", "phase": str(phase), "error": str(error),
               "generated_runs": int(generated_runs)}
    (destination / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        raise TypeError("Dense masks belong in proposal payloads, not metadata")
    return value


def _scope_metadata(scope: object) -> dict[str, object]:
    return dict(_plain(scope)) if isinstance(scope, Mapping) else {"scope_id": str(scope)}


def observation_snapshot_sha256(observations: np.ndarray) -> str:
    """Hash categorical original observations in bounded slice-sized chunks."""
    array = np.asarray(observations)
    digest = hashlib.sha256(json.dumps(list(array.shape)).encode("ascii"))
    for plane in array:
        digest.update(np.packbits(np.asarray(plane) != 0, bitorder="little").tobytes())
    return digest.hexdigest()


def _buffer_identity(array: np.ndarray) -> tuple[object, ...]:
    return (int(array.__array_interface__["data"][0]), tuple(array.shape),
            str(array.dtype), tuple(array.strides))


def _planning_identity(*, scope: object, pass_index: int, gap_distance: int,
                       min_radius: float, search_angle_deg: float,
                       interpolation_walk_back: int, interpolation_candidates: int,
                       interpolation_passes: int, wrap_axis: bool,
                       upstream_lineage: object, spacing_zyx: tuple[float, float, float],
                       planner_limits: object, view: object) -> str:
    metadata = _scope_metadata(scope)
    settings = dict(scope_id=str(metadata.get("scope_id", "sam")),
                    pass_index=int(pass_index), gap_distance=int(gap_distance),
                    min_radius=float(min_radius), search_angle_deg=float(search_angle_deg),
                    interpolation_walk_back=int(interpolation_walk_back),
                    interpolation_candidates=int(interpolation_candidates),
                    interpolation_passes=int(interpolation_passes), wrap_axis=bool(wrap_axis),
                    upstream_lineage=_plain(upstream_lineage), spacing_zyx=list(spacing_zyx),
                    planner_limits=_plain(planner_limits),
                    view_name=str(getattr(view, "name", view)),
                    physical_view=str(getattr(view, "physical_view_name", "")),
                    angle_deg=float(getattr(view, "tta_angle_deg", 0.0) or 0.0),
                    sam_crop_mode=str(metadata.get("sam_crop_mode", "whole")))
    return hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SamPreparedInterpolationPass:
    """Immutable bounded generation plan, prepared before image/model admission.

    The caller pins the same immutable detector observation buffer until execution
    finishes. All group masks and endpoint silhouettes are independently sealed
    by the planner; no image or model resource is retained here.
    """
    plan: object = field(compare=False, repr=False)
    runs: tuple[object, ...] = field(compare=False, repr=False)
    native_shape: tuple[int, ...]
    observation_buffer_identity: tuple[object, ...]
    observation_snapshot_sha256: str
    settings_sha256: str
    planner_wall_seconds: float
    snapshot_wall_seconds: float
    needed_frames: tuple[int, ...]
    frame_crop_bounds: Mapping[int, tuple[int, int, int, int]] = field(compare=False, repr=False)
    canonical_buffer_identity: tuple[object, ...] | None = None
    disabled: bool = False
    crop_mode: str = "whole"
    tracker_jobs: tuple[object, ...] = field(default=(), compare=False, repr=False)
    tile_inventory: Mapping[int, tuple[object, ...]] = field(default_factory=lambda: MappingProxyType({}), compare=False, repr=False)
    tiling_sha256: str = ""
    tiled_assembly_bytes: int = 0

    @property
    def needs_tracking(self) -> bool:
        return bool(self.tracker_jobs if self.crop_mode == "tiled" else self.runs) and not self.disabled

    @property
    def groups(self) -> tuple[object, ...]:
        return () if self.plan is None else self.plan.groups

    def execution_order(self, worker_count: int = 1) -> tuple[int, ...]:
        """Interleave distinct crop queues on multiple admitted workers."""
        work = self.tracker_jobs if self.crop_mode == "tiled" else self.runs
        if self.crop_mode == "tiled":
            return tuple(index for batch in self.execution_batches(worker_count) for index in batch)
        if int(worker_count) <= 1:
            return tuple(range(len(work)))
        groups = {str(group.group_id): group for group in self.groups}
        queues: dict[tuple[int, ...], list[int]] = {}
        for index, run in enumerate(work):
            crop = tuple(run.tile.crop_bbox_yx) if self.crop_mode == "tiled" else tuple(groups[str(run.group_id)].context_bbox_yx)
            queues.setdefault(crop, []).append(index)
        output = []
        for wave in range(max((len(queue) for queue in queues.values()), default=0)):
            for queue in queues.values():
                if wave < len(queue):
                    output.append(queue[wave])
        return tuple(output)

    def execution_batches(self, worker_count=1):
        """Drain bounded parent cohorts before opening another tiled assembly."""
        if self.crop_mode != "tiled":
            return (self.execution_order(worker_count),)
        from .sam_crop_tiling import MAX_ACTIVE_PARENT_ASSEMBLIES
        groups = {str(group.group_id): group for group in self.groups}
        parent_queues = {}
        for index, run in enumerate(self.runs):
            parent_queues.setdefault(tuple(groups[str(run.group_id)].context_bbox_yx), []).append(index)
        parent_order = tuple(range(len(self.runs))) if int(worker_count)<=1 else tuple(
            queue[wave] for wave in range(max((len(q) for q in parent_queues.values()), default=0))
            for queue in parent_queues.values() if wave < len(queue))
        by_parent = {}
        for index, job in enumerate(self.tracker_jobs):
            by_parent.setdefault(job.original_run_index, []).append(index)
        batches = []
        for offset in range(0, len(parent_order), MAX_ACTIVE_PARENT_ASSEMBLIES):
            parents = parent_order[offset:offset+MAX_ACTIVE_PARENT_ASSEMBLIES]
            indices = [index for parent in parents for index in by_parent.get(parent, ())]
            if int(worker_count)>1:
                queues = {}
                for index in indices:
                    queues.setdefault(self.tracker_jobs[index].tile.crop_bbox_yx, []).append(index)
                indices = [queue[wave] for wave in range(max((len(q) for q in queues.values()), default=0))
                           for queue in queues.values() if wave < len(queue)]
            batches.append(tuple(indices))
        return tuple(batches)


def prepare_sam_interpolation_pass(
    observation_volume: np.ndarray, *, view: object = None, scope: object = "sam",
    pass_index: int = 1, gap_distance: int = 15, min_radius: float = 3.0,
    search_angle_deg: float = 15.0, interpolation_walk_back: int = 1,
    interpolation_candidates: int = 1, interpolation_passes: int = 1,
    wrap_axis: bool = False, upstream_lineage: object = None,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    planner_limits: object = None, canonical_labels: np.ndarray | None = None,
    policy: object = None,
    crop_mode: str | None = None,
) -> SamPreparedInterpolationPass:
    """Plan exhaustive original-anchor jobs without images, CUDA, or a tracker."""
    observations = np.asarray(observation_volume)
    if int(gap_distance) <= 0:
        return SamPreparedInterpolationPass(None, (), tuple(observations.shape),
            _buffer_identity(observations), "", "", 0., 0., (), MappingProxyType({}), disabled=True)
    if int(pass_index) < 1 or int(interpolation_passes) < int(pass_index):
        raise ValueError("SAM pass_index must be within interpolation_passes")
    if observations.ndim != 3 or any(value <= 0 for value in observations.shape):
        raise ValueError("SAM interpolation needs a nonempty native TYX observation volume")
    metadata = _scope_metadata(scope)
    from .sam_crop_tiling import resolve_sam_crop_mode
    mode = resolve_sam_crop_mode(crop_mode if crop_mode is not None else metadata.get("sam_crop_mode"))
    metadata["sam_crop_mode"] = mode
    _validate_view(view, wrap_axis, metadata)
    settings_sha256 = _planning_identity(scope=metadata, view=view, pass_index=pass_index,
        gap_distance=gap_distance, min_radius=min_radius, search_angle_deg=search_angle_deg,
        interpolation_walk_back=interpolation_walk_back, interpolation_candidates=interpolation_candidates,
        interpolation_passes=interpolation_passes, wrap_axis=wrap_axis, upstream_lineage=upstream_lineage,
        spacing_zyx=spacing_zyx, planner_limits=planner_limits)
    from .sam_bridge_planning import plan_sam_bridges
    planning_started = time.perf_counter()
    plan = plan_sam_bridges(observations,
        interpolation_distance=int(gap_distance), interpolation_candidates=int(interpolation_candidates),
        interpolation_walk_back=int(interpolation_walk_back), interpolation_passes=int(interpolation_passes),
        interpolation_min_radius=float(min_radius), interpolation_search_angle=float(search_angle_deg),
        scope_id=str(metadata.get("scope_id", "sam")), spacing_zyx=spacing_zyx,
        canonical_labels=canonical_labels, observation_lineage=upstream_lineage,
        limits=planner_limits, planning_pass_index=int(pass_index))
    planning_seconds = time.perf_counter() - planning_started
    runs = tuple(run for run in plan.runs if int(run.pass_index) == int(pass_index))
    if runs:
        # These are runtime/measurement capacity checks, not quality selection.
        # Reject a request before rendering/model admission instead of silently
        # shortening a declared endpoint session or omitting a hypothesis.
        from .lta_sam import SamSessionPlan
        for run in runs:
            try:
                SamSessionPlan(sequence_id=str(run.run_id), session_index=0,
                    frame_start=min(run.expected_frames), frame_stop=max(run.expected_frames) + 1)
            except (TypeError, ValueError) as error:
                raise SamInterpolationInfrastructureError(
                    f"SAM planned run {run.run_id} is outside the bounded tracker session contract: {error}") from error
        from .sam_policy import resolve_sam_bridge_policy
        source_policy = policy or {}
        if "kind" in source_policy and "mode" not in source_policy:
            source_policy = {"sam_bridge_policy": source_policy}
        topology_limit = int(resolve_sam_bridge_policy(source_policy, generation_mode=mode)["max_group_bytes"])
        tracked_group_ids = {str(run.group_id) for run in runs}
        for group in plan.groups:
            if str(group.group_id) not in tracked_group_ids:
                continue
            y0, x0, y1, x1 = group.context_bbox_yx
            if len(group.frame_indices) * (y1 - y0) * (x1 - x0) * 16 > topology_limit:
                raise SamInterpolationInfrastructureError(
                    f"SAM group {group.group_id} topology exceeds its declared memory budget before tracker admission")
    snapshot_started = time.perf_counter()
    snapshot = "" if int(pass_index) > 1 else observation_snapshot_sha256(observations)
    snapshot_seconds = time.perf_counter() - snapshot_started
    tracker_jobs, tile_inventory, tiling_sha256, assembly_bytes = (), MappingProxyType({}), "", 0
    needed_frames, frame_crop_bounds = plan.needed_frames, plan.frame_crop_bounds
    if mode == "tiled":
        from .sam_crop_tiling import prepare_tiled_jobs
        try:
            tracker_jobs, tile_inventory, tiling_sha256, assembly_bytes = prepare_tiled_jobs(runs,
                {str(group.group_id): group for group in plan.groups}, plan.by_id)
        except (MemoryError, ValueError) as error:
            raise SamInterpolationInfrastructureError(f"SAM tiled generation preflight failed: {error}") from error
        needed_frames = tuple(sorted({int(frame) for job in tracker_jobs for frame in job.original_run.expected_frames}))
        demand = {}
        for job in tracker_jobs:
            crop = job.tile.crop_bbox_yx
            for frame in job.original_run.expected_frames:
                old = demand.get(int(frame))
                demand[int(frame)] = crop if old is None else (
                    min(old[0], crop[0]), min(old[1], crop[1]), max(old[2], crop[2]), max(old[3], crop[3]))
        frame_crop_bounds = MappingProxyType(demand)
    return SamPreparedInterpolationPass(plan, runs, tuple(observations.shape),
        _buffer_identity(observations), snapshot, settings_sha256, planning_seconds,
        snapshot_seconds, needed_frames, frame_crop_bounds,
        canonical_buffer_identity=(None if canonical_labels is None else
                                   _buffer_identity(np.asarray(canonical_labels))),
        crop_mode=mode, tracker_jobs=tracker_jobs, tile_inventory=tile_inventory,
        tiling_sha256=tiling_sha256, tiled_assembly_bytes=assembly_bytes)


def _tracker_requests(runs: tuple[object, ...], groups: Mapping[str, object],
                      observations: Mapping[str, object], cancel_event: object):
    """Lazily build one seed payload per fixed run; tracker bounds admission."""
    for run in runs:
        _check_cancelled(cancel_event)
        group = groups[str(run.group_id)]
        y0, x0, y1, x1 = group.context_bbox_yx
        seed_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for seed_id in run.seed_ids:
            seed = observations[str(seed_id)]
            if int(seed.frame_index) != int(run.expected_frames[0]):
                raise SamInterpolationInfrastructureError("SAM seed lineage and injected frame differ")
            seed_mask |= seed.mask_in_crop(group.context_bbox_yx)
        if not np.any(seed_mask):
            raise SamInterpolationInfrastructureError("SAM original-observation seed is empty")
        yield dict(run_id=str(run.run_id), seed_mask=seed_mask,
            seed_frame=int(run.expected_frames[0]), frame_start=min(run.expected_frames),
            frame_stop=max(run.expected_frames) + 1,
            direction="forward" if int(run.direction) > 0 else "backward",
            crop_xyxy=(x0, y0, x1, y1))
        del seed_mask


def _tiled_tracker_requests(jobs, groups, observations, cancel_event):
    from .sam_crop_tiling import clipped_seed_mask
    for job in jobs:
        _check_cancelled(cancel_event)
        run, tile = job.original_run, job.tile
        seed = clipped_seed_mask(run, observations, tile.crop_bbox_yx)
        if int(np.count_nonzero(seed)) != int(tile.seed_foreground) or not seed.any():
            raise SamInterpolationInfrastructureError("SAM tile original seed differs from its sealed generation plan")
        y0, x0, y1, x1 = tile.crop_bbox_yx
        yield dict(run_id=job.run_id, seed_mask=seed, seed_frame=int(run.expected_frames[0]),
            frame_start=min(run.expected_frames), frame_stop=max(run.expected_frames)+1,
            direction="forward" if int(run.direction) > 0 else "backward", crop_xyxy=(x0, y0, x1, y1),
            metadata=dict(parent_run_id=str(run.run_id), tile_id=tile.tile_id,
                ownership_bbox_yx=list(tile.ownership_bbox_yx),
                whole_crop_bbox_yx=list(groups[str(run.group_id)].context_bbox_yx),
                crop_mode="tiled", rule="1008/128/midpoint-v1"))
        del seed


def _iterate_tracker_results(runtime: object, requests: object, cache_ref: object):
    if hasattr(runtime, "iter_results"):
        yield from runtime.iter_results(requests, source_cache_ref=cache_ref)
    else:
        for index, request in enumerate(requests):
            yield index, runtime.run(**request)


def _iterate_tiled_tracker_results(runtime, prepared, worker_count, groups, observations, cache_ref, cancel_event):
    offset = 0
    for batch in prepared.execution_batches(worker_count):
        jobs = tuple(prepared.tracker_jobs[index] for index in batch)
        stream = _iterate_tracker_results(runtime,
            _tiled_tracker_requests(jobs, groups, observations, cancel_event), cache_ref)
        try:
            for index, result in stream:
                valid = not isinstance(index, bool) and isinstance(index, (int, np.integer)) and 0<=int(index)<len(batch)
                yield offset+int(index) if valid else -1, result
        finally:
            stream.close()
        offset += len(batch)


def _store_generated_parent_run(writer, run, result, group, observation_by_id,
                                metadata, upstream_lineage, availability_masks=None):
    descriptor = _run_descriptor(run, result)
    for identity_key in ("sam_model", "sam_runtime"):
        actual_identity = descriptor["runtime_receipt"].get(identity_key)
        if actual_identity is not None:
            if identity_key in metadata and metadata[identity_key] != actual_identity:
                raise SamInterpolationInfrastructureError("SAM model/runtime identity changed within a provenance scope")
            metadata[identity_key] = actual_identity
            writer.scope[identity_key] = actual_identity
    descriptor["lineage"] = {
        "original_detector_observations": [
            {"observation_id": str(identifier), "lineage": _plain(observation_by_id[str(identifier)].lineage)}
            for identifier in tuple(run.seed_ids) + tuple(run.held_out_ids)],
        "upstream": _plain(upstream_lineage)}
    raw_masks = {int(frame): np.asarray(mask) for frame, mask in result.frames.items()}
    branch_masks = [_frame_masks(group.edge_write_masks[edge_id], tuple(group.frame_indices))
                    for edge_id in getattr(run, "edge_ids", ())
                    if edge_id in getattr(group, "edge_write_masks", {})]
    candidates = None
    if branch_masks:
        candidates = {}
        y0, x0, y1, x1 = group.context_bbox_yx
        for frame, mask in raw_masks.items():
            write = np.zeros((y1-y0, x1-x0), dtype=bool)
            for branch in branch_masks:
                write |= branch[frame]
            candidates[frame] = (mask != 0) & write
    if availability_masks is None:
        writer.add_run(descriptor, raw_masks, candidate_masks=candidates)
    else:
        descriptor["sam_crop_mode"] = "tiled"
        writer.add_run(descriptor, raw_masks, candidate_masks=candidates, availability_masks=availability_masks)
    if not bool(descriptor["structurally_valid"]):
        raise SamInterpolationInfrastructureError("SAM tracker produced structurally invalid expected object evidence")
    return descriptor


def _base_stats(pass_index: int, requested_passes: int) -> dict[str, object]:
    return {
        "interpolation_backend": "sam", "backend": "sam",
        "pass_index": int(pass_index), "requested_passes": int(requested_passes),
        "completed_passes": 0, "skipped_passes": 0,
        "num_objects": 0, "num_endpoints": 0,
        "candidate_connections": 0, "accepted_connections": 0,
        "default_bridges": 0, "walk_back_bridges": 0,
        "skipped_by_min_radius": 0, "added_voxels": 0,
        "skipped": False, "wrap_axis": False,
        "endpoint_method": "original_observed_slice_components",
        "planning_backend": "bounded_sam_family_graph",
        "interpolation_render_backend": "mask_conditioned_sam",
        "interpolation_radius_backend": "raw_sam_bridge_cross_sections",
        "bridge_component_count": 0,
    }


def _validate_view(view: object, wrap_axis: bool, scope: Mapping[str, object]) -> None:
    if wrap_axis:
        raise ValueError("SAM interpolation does not support wrapping view frames")
    if float(scope.get("angle_deg", 0.0) or 0.0) != 0.0:
        raise ValueError("SAM interpolation currently requires augmentation angle zero")
    if view is None:
        return
    family = str(getattr(view, "family", "orthogonal")).lower()
    orientation = str(getattr(view, "physical_view_name", "") or
                      getattr(view, "name", getattr(view, "summary_family", view))).lower()
    if family != "orthogonal" or orientation != "transverse":
        raise ValueError("SAM interpolation currently supports native Transverse only")
    if float(getattr(view, "tilt_angle_deg", 0.0) or 0.0) != 0.0:
        raise ValueError("SAM interpolation does not support tilted view geometry")
    if float(getattr(view, "tta_angle_deg", 0.0) or 0.0) != 0.0:
        raise ValueError("SAM interpolation currently requires augmentation angle zero")


def _frame_masks(masks: object, frames: tuple[int, ...]) -> dict[int, np.ndarray]:
    if isinstance(masks, Mapping):
        return {int(frame): np.asarray(mask, dtype=bool) for frame, mask in masks.items()}
    array = np.asarray(masks, dtype=bool)
    if array.ndim != 3 or len(array) != len(frames):
        raise ValueError("SAM plan masks must correspond exactly to group frame_indices")
    return {int(frame): array[index] for index, frame in enumerate(frames)}


def _write_group(writer: object, group: object, observations: Mapping[str, object],
                 min_radius: float) -> None:
    frames = tuple(int(frame) for frame in group.frame_indices)
    bbox = tuple(int(value) for value in group.context_bbox_yx)
    if str(group.status) in {"unresolved", "incomplete", "invalid"}:
        # Capped groups intentionally have no allocated crop contracts. Persist
        # their bounds/reasons without allocating the very resource they exceed.
        writer.add_group({
            "group_id": str(group.group_id), "context_bbox_yx": bbox,
            "frame_indices": frames, "endpoints": [
                {"observation_id": str(observations[str(identifier)].observation_id),
                 "frame_index": int(observations[str(identifier)].frame_index),
                 "canonical_label": int(observations[str(identifier)].canonical_label),
                 "bbox_yx": list(observations[str(identifier)].bbox_yx),
                 "lineage": _plain(observations[str(identifier)].lineage)}
                for identifier in group.observation_ids],
            "edges": [_plain(edge) for edge in group.edges],
            "status": str(group.status), "reasons": list(group.reasons),
            "complete": False, "interpolation_min_radius": float(min_radius),
            "connectivity": 6, "endpoint_identity_basis": "slice_connected_components",
        }, {f"endpoint_local:{identifier}": observations[str(identifier)].mask_crop
            for identifier in group.observation_ids})
        return
    acceptance = _frame_masks(group.acceptance_masks, frames)
    known = _frame_masks(group.known_foreground_masks, frames) if hasattr(group, "known_foreground_masks") else {}
    evaluations = getattr(group, "branch_evaluation_masks", {})
    endpoints = []
    masks: dict[str, np.ndarray] = {}
    for observation_id in group.observation_ids:
        observation = observations[str(observation_id)]
        endpoints.append({
            "observation_id": str(observation.observation_id),
            "frame_index": int(observation.frame_index),
            "canonical_label": int(observation.canonical_label),
            "lineage": _plain(observation.lineage),
        })
        silhouette = observation.mask_in_crop(bbox)
        masks[f"endpoint:{observation_id}"] = silhouette
        # Nonterminal continuation/walk-back observations still need portable
        # references. Only held-out terminal domains are scored by the policy.
        masks[f"evaluation:{observation_id}"] = np.asarray(
            evaluations.get(str(observation_id), acceptance[int(observation.frame_index)]), dtype=bool)
        if int(observation.frame_index) in known:
            permitted = getattr(group, "branch_permitted_masks", {}).get(str(observation_id))
            masks[f"permitted:{observation_id}"] = (
                np.asarray(permitted, dtype=bool) if permitted is not None else
                known[int(observation.frame_index)] & ~silhouette)
    for frame, mask in acceptance.items():
        masks[f"acceptance:{frame}"] = mask
    for frame, mask in _frame_masks(group.write_masks, frames).items():
        masks[f"write:{frame}"] = mask
    if hasattr(group, "unrelated_masks"):
        for frame, mask in _frame_masks(group.unrelated_masks, frames).items():
            masks[f"unrelated:{frame}"] = mask
    for frame, mask in known.items():
        masks[f"known_foreground:{frame}"] = mask
    for edge_id, masks_by_frame in getattr(group, "edge_write_masks", {}).items():
        for frame, mask in _frame_masks(masks_by_frame, frames).items():
            masks[f"edge_write:{edge_id}:{frame}"] = mask
    for edge_id, masks_by_frame in getattr(group, "edge_contract_masks", {}).items():
        for frame, mask in _frame_masks(masks_by_frame, frames).items():
            masks[f"edge_contract:{edge_id}:{frame}"] = mask
    metadata = {
        "group_id": str(group.group_id), "context_bbox_yx": bbox,
        "frame_indices": frames, "endpoints": endpoints,
        "edges": [_plain(edge) for edge in group.edges],
        "status": str(group.status), "reasons": _plain(group.reasons),
        "complete": str(group.status) not in {"incomplete", "unresolved", "invalid"},
        "interpolation_min_radius": float(min_radius),
        "connectivity": 6,
        "endpoint_identity_basis": "slice_connected_components",
        "endpoint_ids": list(getattr(group, "endpoint_ids", tuple(evaluations))),
    }
    writer.add_group(metadata, masks)


def _run_descriptor(run: object, result: object) -> dict[str, object]:
    frames = {int(frame): mask for frame, mask in result.frames.items()}
    expected = tuple(int(frame) for frame in run.expected_frames)
    receipt = dict(getattr(result, "receipt", {}) or {})
    complete = all(frame in frames for frame in expected)
    complete = complete and bool(receipt.get("coverage_complete", receipt.get("complete", True)))
    structurally_valid = bool(receipt.get("prediction_valid", True))
    return {
        "run_id": str(run.run_id), "group_id": str(run.group_id),
        "backend": "sam", "seed_ids": list(run.seed_ids),
        "held_out_ids": list(run.held_out_ids), "direction": int(run.direction),
        "edge_ids": list(getattr(run, "edge_ids", ())),
        "expected_frames": expected, "observed_frames": sorted(frames),
        "injected_frames": [int(expected[0])],
        "pass_index": int(run.pass_index),
        "walk_back_index": int(run.walk_back_index),
        "complete": complete,
        "structurally_valid": structurally_valid,
        "status": ("infrastructure_invalid" if not structurally_valid else
                   "generated_complete" if complete else "generated_incomplete"),
        "tracker_scores": _plain(getattr(result, "tracker_scores", {})),
        "observation_status": _plain(getattr(result, "observation_status", {})),
        "runtime_receipt": _plain(receipt),
    }


def _records_by_id(records: object, key: str) -> dict[str, Mapping[str, object]]:
    if isinstance(records, Mapping):
        return {str(identifier): record for identifier, record in records.items()}
    return {str(record[key]): record for record in records}


def selected_sam_plane(bundle: object, receipt: Mapping[str, object], frame: int,
                       shape_yx: tuple[int, int], *, direction: int | None = None) -> np.ndarray:
    """Rebuild a selected native plane from owners, preserving shared voxels."""
    from .sam_mask_reader import effective_candidate_mask
    groups = _records_by_id(bundle.groups, "group_id")
    runs = _records_by_id(bundle.runs, "run_id")
    plane = np.zeros(tuple(shape_yx), dtype=np.uint8)
    for run_id in receipt.get("selected_run_ids", ()):
        run = runs[str(run_id)]
        run_direction = run["direction"]
        sign = (1 if run_direction == "forward" else -1) if isinstance(run_direction, str) else int(run_direction)
        if direction is not None and sign != int(direction):
            continue
        if int(frame) not in {int(value) for value in run["expected_frames"]}:
            continue
        group = groups[str(run["group_id"])]
        y0, x0, y1, x1 = (int(value) for value in group["context_bbox_yx"])
        crop = effective_candidate_mask(bundle, str(run_id), int(frame), receipt)
        if crop is None:
            continue
        crop = np.asarray(crop, dtype=bool)
        if crop.shape != (y1 - y0, x1 - x0):
            raise SamInterpolationInfrastructureError("Candidate mask shape differs from stored crop geometry")
        if not (0 <= y0 < y1 <= shape_yx[0] and 0 <= x0 < x1 <= shape_yx[1]):
            raise SamInterpolationInfrastructureError("Candidate crop is outside its native view canvas")
        plane[y0:y1, x0:x1] |= crop
    return plane


def _publish_directions(bundle: object, receipt: Mapping[str, object], observations: np.ndarray,
                        destination: Path, metadata: Mapping[str, object],
                        pass_index: int, merged_work_dir: Path,
                        cancel_event: object = None) -> tuple[np.ndarray, list[dict[str, object]], int, dict[str, object]]:
    merged = None
    try:
        with bundle.reader(max_cache_bytes=32 * 1024**2) as reader:
            selection_view = {**receipt, "mask_filter": reader.filter_snapshot(receipt)}
            merged, components, added = _publish_directions_with_reader(reader, receipt,
                observations, destination, metadata, pass_index, merged_work_dir,
                cancel_event, selection_view)
        return merged, components, added, dict(reader.stats)
    except BaseException:
        if isinstance(merged, np.memmap):
            merged._mmap.close()
        raise


def _publish_directions_with_reader(bundle: object, receipt: Mapping[str, object], observations: np.ndarray,
                        destination: Path, metadata: Mapping[str, object],
                        pass_index: int, merged_work_dir: Path,
                        cancel_event: object = None,
                        selection_view: Mapping[str, object] | None = None) -> tuple[np.ndarray, list[dict[str, object]], int]:
    # Ordinary binary cvol readers remain unchanged. Only the proposal bundle
    # carries multiple overlapping owners; output unions are reconstructed here.
    from .interpolation import INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter

    shape = tuple(int(value) for value in observations.shape)
    merged_path = merged_work_dir / "selected_merged.u8.dat"
    merged = None
    added = 0
    components = []
    groups = _records_by_id(bundle.groups, "group_id")
    runs = _records_by_id(bundle.runs, "run_id")
    selection_path = destination / "selection.json"
    try:
        for direction, sign in (("forward", 1), ("backward", -1)):
            _check_cancelled(cancel_event)
            directional_run_ids = [str(identifier) for identifier in receipt.get("selected_run_ids", ())
                                   if runs[str(identifier)]["direction"] in (sign, direction)]
            active_frames = sorted({int(frame) for identifier in directional_run_ids
                                    for frame, key in runs[identifier]["candidate_mask_keys"].items()
                                    if int(bundle.records[key]["foreground"]) > 0})
            path = destination / f"sam_bridge_pass{pass_index:02d}_{direction}.cvol"
            writer = IncrementalRawBBoxMaskStoreWriter(
                shape=shape, store_dir=path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                desc=f"Selected SAM {direction} pass {pass_index}",
                extra_meta={**dict(metadata), "interpolation_backend": "sam",
                            "interpolation_direction": direction,
                            "interpolation_pass_index": int(pass_index),
                            "sam_policy_hash": str(receipt["policy_hash"]),
                            "sam_mask_filter": _plain(receipt.get("mask_filter")),
                            "sam_evidence_path": str(bundle.directory)},
            )
            try:
                next_unwritten = 0
                for frame in active_frames:
                    _check_cancelled(cancel_event)
                    if frame > next_unwritten:
                        writer.consume_empty_range(next_unwritten, frame - next_unwritten)
                    plane = selected_sam_plane(bundle, selection_view or receipt, frame, shape[1:], direction=sign)
                    # The write contract is additive. Guard this invariant again
                    # at publication in case a corrupt/custom evidence writer errs.
                    if np.any(plane & (np.asarray(observations[frame]) != 0)):
                        raise SamInterpolationInfrastructureError("SAM candidate repaints original observations")
                    writer.consume(frame, plane[None])
                    next_unwritten = frame + 1
                    if plane.any():
                        if merged is None:
                            merged_work_dir.mkdir(parents=True, exist_ok=True)
                            merged = np.memmap(merged_path, mode="w+", dtype=np.uint8, shape=shape)
                            snapshot = hashlib.sha256(json.dumps(list(shape)).encode("ascii"))
                            for source_frame in range(shape[0]):
                                _check_cancelled(cancel_event)
                                original_plane = np.asarray(observations[source_frame]) != 0
                                merged[source_frame] = original_plane
                                snapshot.update(np.packbits(original_plane, bitorder="little").tobytes())
                            expected_snapshot = str(metadata.get("observation_snapshot_sha256", ""))
                            if expected_snapshot and snapshot.hexdigest() != expected_snapshot:
                                raise SamInterpolationInfrastructureError("Pinned SAM original observations changed during tracking")
                        added += int(np.count_nonzero(plane & ~(merged[frame] != 0)))
                        merged[frame] |= plane
                if next_unwritten < shape[0]:
                    writer.consume_empty_range(next_unwritten, shape[0] - next_unwritten)
                _check_cancelled(cancel_event)
                store_meta = writer.finalize()
            except BaseException as error:
                writer.abort(error)
                raise
            components.append({
                "direction": direction, "path": str(path),
                "storage_format": INTERNAL_PACKED_CVOL_FORMAT,
                "voxel_count": int(store_meta.get("foreground_voxels", store_meta.get("added_voxels", 0))),
                "evidence_path": str(bundle.directory),
                "policy_hash": str(receipt["policy_hash"]), "metadata": store_meta,
                "selection_receipt_path": str(selection_path),
            })
            directional_group_ids = sorted({str(runs[identifier]["group_id"]) for identifier in directional_run_ids})
            observation_roots = sorted({str(identifier) for run_id in directional_run_ids
                                        for key in ("seed_ids", "held_out_ids")
                                        for identifier in runs[run_id].get(key, ())})
            group_receipts = receipt.get("group_receipts", {})
            all_connected = bool(directional_group_ids) and all(
                group_receipts.get(group_id, {}).get("topology", {}).get("all_requested_edges_connected", False)
                for group_id in directional_group_ids)
            components[-1].update(
                run_ids=directional_run_ids, group_ids=directional_group_ids,
                observation_roots=observation_roots,
                connection_status="connected" if all_connected else "not_connected_or_not_assessed",
                connection_assessment_scope="complete_selected_group_union",
                topology_connectivity=int(receipt.get("resolved_policy", {}).get("connectivity", 6)))
        if merged is not None:
            merged.flush()
        return observations if merged is None else merged, components, added
    except BaseException:
        if merged is not None:
            merged.flush()
            merged._mmap.close()
        raise


def interpolate_sam_view_volume_pass(
    observation_volume: np.ndarray, *, image_provider: object = None, view: object = None,
    work_dir: Path, pass_index: int = 1, gap_distance: int = 15, min_radius: float = 3.0,
    search_angle_deg: float = 15.0, interpolation_walk_back: int = 1,
    interpolation_candidates: int = 1, workers: int = 1, wrap_axis: bool = False,
    runtime: object = None, scope: object = "sam", upstream_lineage: object = None,
    return_bridge_components: bool = False, interpolation_passes: int = 1,
    policy: object = None, spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    planner_limits: object = None, canonical_labels: np.ndarray | None = None,
    runtime_work_dir: Path | None = None,
    cancel_event: object = None,
    prepared_plan: SamPreparedInterpolationPass | None = None,
    crop_mode: str | None = None,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    """Generate one SAM planning round from immutable detector observations.

    ``runtime`` is a persistent raw-mask tracker. ``image_provider`` is its
    caller-owned immutable native uint8 cache reference; no images survive in
    this function after a bounded independent run. Never pass a bridge-mutated
    volume as ``observation_volume``. The returned merged mmap is caller-owned.
    """
    del workers  # Tracking concurrency is admitted and owned by the persistent runtime.
    started = time.perf_counter()
    stats = _base_stats(pass_index, interpolation_passes)
    if int(gap_distance) <= 0:
        stats.update(skipped=True, skip_reason="interpolation_disabled", inactive_configured_backend="sam")
        return observation_volume, stats, []
    if int(pass_index) < 1 or int(interpolation_passes) < int(pass_index):
        raise ValueError("SAM pass_index must be within interpolation_passes")
    observations = np.asarray(observation_volume)
    if observations.ndim != 3 or any(value <= 0 for value in observations.shape):
        raise ValueError("SAM interpolation needs a nonempty native TYX observation volume")
    metadata = _scope_metadata(scope)
    from .sam_crop_tiling import resolve_sam_crop_mode
    resolved_mode = resolve_sam_crop_mode(crop_mode if crop_mode is not None else
        metadata.get("sam_crop_mode", prepared_plan.crop_mode if prepared_plan is not None else None))
    metadata["sam_crop_mode"] = resolved_mode
    _validate_view(view, wrap_axis, metadata)

    from .sam_evidence import SamEvidenceWriter
    from .sam_policy import select_sam_proposals

    if prepared_plan is None:
        prepared_plan = prepare_sam_interpolation_pass(observation_volume, view=view, scope=metadata,
            pass_index=pass_index, gap_distance=gap_distance, min_radius=min_radius,
            search_angle_deg=search_angle_deg, interpolation_walk_back=interpolation_walk_back,
            interpolation_candidates=interpolation_candidates, interpolation_passes=interpolation_passes,
            wrap_axis=wrap_axis, upstream_lineage=upstream_lineage, spacing_zyx=spacing_zyx,
            planner_limits=planner_limits, canonical_labels=canonical_labels, policy=policy, crop_mode=resolved_mode)
    else:
        expected_settings = _planning_identity(scope=metadata, view=view, pass_index=pass_index,
            gap_distance=gap_distance, min_radius=min_radius, search_angle_deg=search_angle_deg,
            interpolation_walk_back=interpolation_walk_back, interpolation_candidates=interpolation_candidates,
            interpolation_passes=interpolation_passes, wrap_axis=wrap_axis, upstream_lineage=upstream_lineage,
            spacing_zyx=spacing_zyx, planner_limits=planner_limits)
        if (prepared_plan.disabled or prepared_plan.settings_sha256 != expected_settings
                or prepared_plan.observation_buffer_identity != _buffer_identity(observations)
                or prepared_plan.canonical_buffer_identity != (None if canonical_labels is None else
                                                              _buffer_identity(np.asarray(canonical_labels)))):
            raise ValueError("Prepared SAM plan differs from its pinned original observations or planning settings")
        if (observations.flags.writeable and prepared_plan.observation_snapshot_sha256
                and observation_snapshot_sha256(observations) != prepared_plan.observation_snapshot_sha256):
            raise ValueError("Prepared SAM original observation snapshot changed before execution")
    plan = prepared_plan.plan
    runs = prepared_plan.runs
    stats["planner_wall_seconds"] = prepared_plan.planner_wall_seconds
    stats["observation_snapshot_wall_seconds"] = prepared_plan.snapshot_wall_seconds
    stats["sam_prepared_needed_frames"] = list(prepared_plan.needed_frames)
    stats["sam_prepared_frame_count"] = len(prepared_plan.needed_frames)
    stats["sam_crop_mode"] = resolved_mode
    if resolved_mode == "tiled":
        from .sam_crop_tiling import tiling_recipe, TILE_EVIDENCE_SCHEMA
        metadata.update(sam_tiling_recipe=tiling_recipe(), tile_evidence_schema=TILE_EVIDENCE_SCHEMA,
                        sam_tiling_plan_sha256=prepared_plan.tiling_sha256)
        metadata["sam_tiled_plan"] = [
            dict(parent_run_id=str(runs[index].run_id), group_id=str(runs[index].group_id),
                expected_frames=list(runs[index].expected_frames), seed_ids=list(runs[index].seed_ids),
                tiles=[dict(tile_id=tile.tile_id, child_run_id=f"{runs[index].run_id}__{tile.tile_id}",
                    crop_bbox_yx=list(tile.crop_bbox_yx), ownership_bbox_yx=list(tile.ownership_bbox_yx),
                    seed_foreground=int(tile.seed_foreground), seed_available=tile.attempted,
                    planning_status="planned_original_seed_tile" if tile.attempted else "unavailable_empty_original_seed")
                    for tile in tiles])
            for index, tiles in prepared_plan.tile_inventory.items()]
        stats.update(sam_tiled_original_runs=len(runs), sam_tiled_child_jobs=len(prepared_plan.tracker_jobs),
            sam_tiled_groups=len({str(run.group_id) for run in runs}),
            sam_tiled_multi_tile_runs=sum(len(tiles)>1 for tiles in prepared_plan.tile_inventory.values()),
            sam_tiled_skipped_empty_seed_tiles=sum(not tile.attempted for tiles in prepared_plan.tile_inventory.values() for tile in tiles),
            sam_tiled_assembly_logical_bytes=int(prepared_plan.tiled_assembly_bytes))
    stats.update(num_objects=len(plan.observations), num_endpoints=len(plan.observations),
                 candidate_connections=sum(len(group.edges) for group in plan.groups),
                 planner_plan_count=len(runs), requested_passes=int(plan.requested_passes),
                 completed_passes=int(plan.completed_passes), skipped_passes=int(plan.skipped_passes))
    stats["sam_planning_status"] = str(plan.status)
    stats["sam_planning_reasons"] = list(plan.reasons)
    stats["sam_group_planning_receipts"] = [
        {"group_id": str(group.group_id), "status": str(group.status),
         "reasons": list(group.reasons)} for group in plan.groups]
    unresolved = str(plan.status) == "unresolved" or any(str(group.status) == "unresolved" for group in plan.groups)
    stats["sam_unresolved_groups"] = sum(str(group.status) == "unresolved" for group in plan.groups)
    if not runs and int(pass_index) > 1:
        stats.update(skipped=True, skip_reason="observed_anchor_hypotheses_exhausted",
                     generator_wall_seconds=time.perf_counter() - started)
        return observation_volume, stats, []
    if not runs:
        stats.update(skipped=False, planning_no_jobs=True,
                     skip_reason="resource_limit_unresolved" if unresolved else "no_missing_connections")
    if runs and runtime is None:
        raise ValueError("Active SAM interpolation requires a persistent raw-mask tracker runtime")
    cache_ref = getattr(image_provider, "cache_ref", image_provider)
    if cache_ref is not None:
        if tuple(int(value) for value in cache_ref.shape) != observations.shape:
            raise ValueError("SAM native image cache and observation canvas geometry differ")
        if hasattr(cache_ref, "revalidate"):
            cache_ref.revalidate()
        if not hasattr(runtime, "iter_results") and hasattr(runtime, "set_source_cache"):
            runtime.set_source_cache(cache_ref)

    metadata.update({
        "backend": "sam", "shape_tyx": list(observations.shape),
        "observation_snapshot_sha256": prepared_plan.observation_snapshot_sha256,
        "upstream_lineage": _plain(upstream_lineage),
        "image_snapshot_sha256": str(getattr(cache_ref, "identity_sha256", "")),
        "model_identity": str(getattr(runtime, "model_path", "")),
        "pass_index": int(pass_index), "spacing_zyx": list(spacing_zyx),
        "planning_contract": "original_observations_only_fixed_context_branch_write",
        "interpolation_min_radius": float(min_radius),
        "planning_status": str(plan.status), "planning_reasons": list(plan.reasons),
        "selection_receipt_required": True,
    })
    metadata["input_fingerprints"] = {
        "original_snapshot": str(metadata["observation_snapshot_sha256"]),
        "image_snapshot": str(metadata["image_snapshot_sha256"]),
        "planning_inventory": str(getattr(plan, "inventory_fingerprint", "")),
        "planning_contract": str(getattr(plan, "planning_fingerprint", "")),
    }
    if resolved_mode == "tiled":
        metadata["input_fingerprints"]["sam_tiling_plan"] = prepared_plan.tiling_sha256
    if isinstance(upstream_lineage, Mapping):
        for key in ("upstream_fingerprints", "gate_support_fingerprints"):
            if isinstance(upstream_lineage.get(key), Mapping):
                metadata[key] = _plain(upstream_lineage[key])
        gate_identity = str(upstream_lineage.get("gate_support_identity", "") or "")
        if gate_identity:
            metadata.setdefault("gate_support_fingerprints", {})[
                str(upstream_lineage.get("parent_scope", "parent_bridge"))] = gate_identity
    scope_hash = hashlib.sha256(json.dumps(_plain(metadata), sort_keys=True).encode("utf-8")).hexdigest()[:20]
    destination = Path(work_dir) / f"sam_{scope_hash}"
    destination.mkdir(parents=True, exist_ok=True)
    writer = SamEvidenceWriter(destination / "evidence", metadata)
    observation_by_id = {str(observation.observation_id): observation for observation in plan.observations}
    groups = {str(group.group_id): group for group in plan.groups}
    generated_by_index: dict[int, dict[str, object]] = {}
    worker_count = len(getattr(runtime, "device_ids", ())) or 1
    execution_order = prepared_plan.execution_order(worker_count)
    tracking_work = prepared_plan.tracker_jobs if resolved_mode == "tiled" else runs
    execution_runs = tuple(tracking_work[index] for index in execution_order)
    stats["sam_execution_schedule"] = ("bounded_tiled_parent_cohorts" if resolved_mode == "tiled" else
                                      "crop_waves" if worker_count > 1 else "group_contiguous")
    stats["sam_execution_order"] = list(execution_order)
    assemblies = {}
    completed_jobs = set()
    assembly_root = ((Path(runtime_work_dir) / scope_hash) if runtime_work_dir is not None else
                     destination / "_runtime") / "tile_assemblies"
    result_stream = None
    try:
        for group in plan.groups:
            _check_cancelled(cancel_event)
            _write_group(writer, group, observation_by_id, float(min_radius))
        if resolved_mode == "tiled":
            from .sam_crop_tiling import tile_descriptor, TiledRunAssembly, MAX_ACTIVE_PARENT_ASSEMBLIES
            for parent_index, tiles in prepared_plan.tile_inventory.items():
                for tile in tiles:
                    if not tile.attempted:
                        writer.add_run_tile(runs[parent_index].run_id, tile_descriptor(runs[parent_index], tile), {})
        tracker_started = time.perf_counter()
        result_stream = (_iterate_tiled_tracker_results(runtime, prepared_plan, worker_count, groups,
            observation_by_id, cache_ref, cancel_event) if resolved_mode == "tiled" else
            _iterate_tracker_results(runtime, _tracker_requests(execution_runs, groups,
                observation_by_id, cancel_event), cache_ref))
        for execution_index, result in result_stream:
            try:
                if (isinstance(execution_index, bool) or not isinstance(execution_index, (int, np.integer))
                        or not 0 <= int(execution_index) < len(tracking_work)):
                    raise SamInterpolationInfrastructureError("SAM worker returned duplicate or unknown run ownership")
                work_index = execution_order[int(execution_index)]
                if work_index in completed_jobs:
                    raise SamInterpolationInfrastructureError("SAM worker returned duplicate or unknown run ownership")
                job = tracking_work[work_index]
                run = job.original_run if resolved_mode == "tiled" else job
                run_index = job.original_run_index if resolved_mode == "tiled" else work_index
                group = groups[str(run.group_id)]
                expected_run_id = job.run_id if resolved_mode == "tiled" else run.run_id
                declared_run_id = getattr(result, "receipt", {}).get("run_id")
                if declared_run_id is not None and str(declared_run_id) != str(expected_run_id):
                    raise SamInterpolationInfrastructureError("SAM worker result differs from its planned run identity")
                if resolved_mode == "whole":
                    generated_by_index[run_index] = _store_generated_parent_run(writer, run, result,
                        group, observation_by_id, metadata, upstream_lineage)
                else:
                    for identity_key in ("sam_model", "sam_runtime"):
                        actual_identity = dict(result.receipt or {}).get(identity_key)
                        if actual_identity is not None:
                            actual_identity = _plain(actual_identity)
                            if identity_key in metadata and metadata[identity_key] != actual_identity:
                                raise SamInterpolationInfrastructureError("SAM tile model/runtime identity changed within a scope")
                            metadata[identity_key] = actual_identity
                            writer.scope[identity_key] = actual_identity
                    if run_index not in assemblies:
                        if len(assemblies) >= MAX_ACTIVE_PARENT_ASSEMBLIES:
                            raise SamInterpolationInfrastructureError("SAM tiled parent assembly admission exceeded its bounded cohort")
                        assemblies[run_index] = TiledRunAssembly(run, group,
                            prepared_plan.tile_inventory[run_index], assembly_root / str(run.run_id))
                    assembly = assemblies[run_index]
                    child_descriptor, raw_tile_masks = assembly.consume(job, result)
                    writer.add_run_tile(run.run_id, child_descriptor, raw_tile_masks)
                    if not child_descriptor["structurally_valid"]:
                        raise SamInterpolationInfrastructureError("SAM tile tracker produced structurally invalid expected object evidence")
                    if assembly.ready:
                        assembled_result = assembly.result()
                        for identity_key in ("sam_model", "sam_runtime"):
                            if identity_key in metadata:
                                assembled_result.receipt[identity_key] = metadata[identity_key]
                        generated_by_index[run_index] = _store_generated_parent_run(writer, run, assembled_result,
                            group, observation_by_id, metadata, upstream_lineage, assembly.availability())
                        del assembled_result
                        assembly.close()
                        del assemblies[run_index]
                    del raw_tile_masks
                completed_jobs.add(work_index)
            finally:
                if hasattr(runtime, "release_result"):
                    runtime.release_result(result)
                del result
        if len(completed_jobs) != len(tracking_work) or len(generated_by_index) != len(runs):
            raise SamInterpolationInfrastructureError("SAM worker stream omitted expected independently seeded runs/tiles")
        if resolved_mode == "tiled":
            stats["sam_tiled_child_jobs_generated"] = len(completed_jobs)
            stats["sam_tiled_parent_runs_generated"] = len(generated_by_index)
        stats["sam_tracker_wall_seconds"] = time.perf_counter() - tracker_started
        bundle = writer.commit(complete=True)
    except BaseException as error:
        try:
            writer.commit(complete=False)
        except BaseException:
            pass
        _failure_receipt(destination, error, "generation", len(generated_by_index))
        raise SamInterpolationInfrastructureError(f"SAM generation failed; incomplete evidence at {destination}: {error}") from error
    finally:
        original_error = sys.exc_info()[1]
        assembly_cleanup_errors = []
        for assembly in assemblies.values():
            try:
                assembly.close()
            except Exception as assembly_cleanup_error:
                if original_error is not None:
                    add_note = getattr(original_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"SAM tiled assembly cleanup failed: {assembly_cleanup_error}")
                else:
                    assembly_cleanup_errors.append(assembly_cleanup_error)
        if result_stream is not None:
            try:
                result_stream.close()
            except BaseException as cleanup_error:
                if original_error is None:
                    _failure_receipt(destination, cleanup_error, "worker_cleanup", len(generated_by_index))
                    raise SamInterpolationInfrastructureError(
                        f"SAM worker cleanup failed; retained evidence at {destination}: {cleanup_error}") from cleanup_error
                add_note = getattr(original_error, "add_note", None)
                if callable(add_note):
                    add_note(f"SAM worker cleanup also failed: {cleanup_error}")
                failure_path = destination / "failure.json"
                try:
                    failure = json.loads(failure_path.read_text(encoding="utf-8"))
                    failure["worker_cleanup_error"] = str(cleanup_error)
                    failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
                except Exception as receipt_error:
                    add_note = getattr(original_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"SAM cleanup receipt could not be updated: {receipt_error}")
        if assembly_cleanup_errors:
            _failure_receipt(destination, assembly_cleanup_errors[0], "tiled_assembly_cleanup", len(generated_by_index))
            raise SamInterpolationInfrastructureError(
                f"SAM tiled assembly cleanup failed: {assembly_cleanup_errors[0]}") from assembly_cleanup_errors[0]
    generated = [generated_by_index[index] for index in sorted(generated_by_index)]

    policy_started = time.perf_counter()
    merged_directory = ((Path(runtime_work_dir) / scope_hash) if runtime_work_dir is not None
                        else destination / "_runtime")
    try:
        _check_cancelled(cancel_event)
        receipt = select_sam_proposals(bundle, policy=policy)
        stats["sam_policy_wall_seconds"] = time.perf_counter() - policy_started
        (destination / "selection.json").write_text(json.dumps(_plain(receipt), indent=2), encoding="utf-8")
        merged, components, added, publication_cache_stats = _publish_directions(bundle, receipt, observations,
            destination, metadata, int(pass_index), merged_directory, cancel_event)
        stats["sam_publication_mask_reader"] = publication_cache_stats
        if not added:
            if (observations.flags.writeable and prepared_plan.observation_snapshot_sha256
                    and observation_snapshot_sha256(observations) != prepared_plan.observation_snapshot_sha256):
                raise SamInterpolationInfrastructureError("SAM original observations changed during a no-op execution")
            merged = observation_volume
    except BaseException as error:
        _failure_receipt(destination, error, "selection_or_publication", len(generated))
        raise SamInterpolationInfrastructureError(f"SAM selection/publication failed; retained evidence at {destination}: {error}") from error
    selected_ids = set(str(identifier) for identifier in receipt.get("selected_run_ids", ()))
    selected = [run for run in generated if str(run["run_id"]) in selected_ids]
    selected_groups = {str(run["group_id"]) for run in selected}
    stats.update({
        "accepted_connections": sum(len(groups[group_id].edges) for group_id in selected_groups),
        "default_bridges": sum(int(run["walk_back_index"]) == 0 for run in selected),
        "walk_back_bridges": sum(int(run["walk_back_index"]) > 0 for run in selected),
        "added_voxels": int(added), "bridge_component_count": 2,
        "sam_generated_runs": len(generated), "sam_selected_runs": len(selected),
        "sam_incomplete_runs": sum(not bool(run["complete"]) for run in generated),
        "skipped_by_min_radius": 0,
        "sam_mask_filter": _plain(receipt.get("mask_filter")),
        "sam_radius_filter_run_summaries": {
            run_id: _plain(record.get("mask_filter_summary", {}))
            for run_id, record in receipt.get("run_receipts", {}).items()},
        "sam_radius_removed_component_observations": sum(
            int(item.get("removed_component_count", 0))
            for record in receipt.get("run_receipts", {}).values()
            for item in record.get("measurements", {}).get("component_filter", ())),
        "sam_radius_removed_raw_foreground_observations": sum(
            int(item.get("removed_foreground", 0))
            for record in receipt.get("run_receipts", {}).values()
            for item in record.get("measurements", {}).get("component_filter", ())),
        "sam_radius_removed_candidate_observations": sum(
            int(item.get("removed_candidate_foreground", 0))
            for record in receipt.get("run_receipts", {}).values()
            for item in record.get("measurements", {}).get("component_filter", ())),
        "sam_policy_hash": str(receipt["policy_hash"]),
        "sam_evidence_path": str(bundle.directory), "sam_selection_receipt": _plain(receipt),
        "sam_directional_components": components,
        "sam_scope_metadata": metadata,
        "sam_merged_workspace_path": str(merged.filename) if added else "",
        "sam_merged_workspace_temporary": bool(added),
        "sam_merged_workspace_bytes": int(merged.nbytes) if added else 0,
        "generator_wall_seconds": time.perf_counter() - started,
    })
    (destination / "generation.json").write_text(json.dumps(_plain(stats), indent=2), encoding="utf-8")
    return merged, stats, components if return_bridge_components else []


__all__ = ["SamInterpolationInfrastructureError", "SamPreparedInterpolationPass",
           "prepare_sam_interpolation_pass", "interpolate_sam_view_volume_pass",
           "observation_snapshot_sha256", "selected_sam_plane"]
