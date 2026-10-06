"""Observation-only SAM bridge generation and selected directional publication.

This module deliberately imports neither SAM nor the SDF generator at import
time. A tracker generates complete raw proposals; the proposal policy selects
contributors before any additive volume or parent-gate support is published.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass, replace
import hashlib
import heapq
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import MappingProxyType
from typing import Any

import numpy as np


class SamInterpolationInfrastructureError(RuntimeError):
    """A generation failure, which must never become an empty successful bridge."""


@contextmanager
def _trace_sam_phase(phase, scope_id, *, operation=None):
    """Retain host phase boundaries in the existing optional task trace."""
    from .runtime import runtime_trace_event
    fields = dict(sam_phase=str(phase), scope_id=str(scope_id),
                  sam_phase_id=f'{threading.get_ident()}:{time.monotonic_ns()}')
    if operation is not None:
        fields['sam_operation'] = str(operation)
    runtime_trace_event('sam_phase_begin', **fields)
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        runtime_trace_event('sam_phase_end', failed=failed, **fields)


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
    from .sam_bridge_planning import (SAM_CROP_PLANNING_CONTRACT_VERSION,
                                      SAM_CONTRACT_STORAGE_VERSION, SamPlanningLimits)
    from .sam_cyclic import CYCLIC_FRAME_ADDRESSING_SCHEMA, IMPLEMENTATION_SHA256 as CYCLIC_IMPLEMENTATION_SHA256
    metadata = _scope_metadata(scope)
    if view is not None and is_dataclass(view):
        # Match the canonical sampler recipe: presentation, augmentation
        # provenance and certificate diagnostics do not alter image samples.
        # In particular an infinite diagnostic error bound is legitimate.
        diagnostic_fields = {"name", "summary_family", "display_name", "physical_view_name",
            "tta_aug_id", "tta_angle_deg", "augmentation_pass", "augmentation_base_view",
            "sampling_policy", "sampling_certificate", "sampling_error_bound_sq",
            "sampling_reference_frames", "sampling_reason"}
        view_recipe = _plain({key: value for key, value in asdict(view).items() if key not in diagnostic_fields})
    else:
        # Controlled callers can use lightweight view descriptors. Bind every
        # parameter used by established native renderers, not just its name.
        recipe_keys = ("name", "physical_view_name", "family", "num_slices", "src_h", "src_w",
            "pad_mode", "tta_angle_deg", "tilt_angle_deg", "tilt_direction", "tilt_base_view",
            "tilt_frame_start", "tilt_frame_stop", "horizontal_axis", "vertical_axis", "stack_axis",
            "full_t", "full_h", "full_w", "azimuths_deg", "diameter", "center_x", "center_y",
            "roi_radius", "azimuthal_base_view", "azimuthal_tilted_source", "azimuthal_source_view_name",
            "radial_base_view", "radial_tilted_source", "radial_source_view_name", "radial_radii",
            "radial_arc_origin", "radial_height_origin", "radial_patch_size", "radial_patch_index",
            "spherical_face", "spherical_radii", "spherical_face_intervals", "spherical_patch_size",
            "spherical_u_origin", "spherical_v_origin", "spherical_rotation_xyz")
        view_recipe = {key: _plain(getattr(view, key)) for key in recipe_keys if hasattr(view, key)}
    settings = dict(scope_id=str(metadata.get("scope_id", "sam")),
                    crop_contract_version=SAM_CROP_PLANNING_CONTRACT_VERSION,
                    contract_storage_version=SAM_CONTRACT_STORAGE_VERSION,
                    interpolation_session_contract="tta_complete_endpoint_interval_without_lta30",
                    pass_index=int(pass_index), gap_distance=int(gap_distance),
                    min_radius=float(min_radius), search_angle_deg=float(search_angle_deg),
                    interpolation_walk_back=int(interpolation_walk_back),
                    interpolation_candidates=int(interpolation_candidates),
                    interpolation_passes=int(interpolation_passes), wrap_axis=bool(wrap_axis),
                    cyclic_address_schema=CYCLIC_FRAME_ADDRESSING_SCHEMA if wrap_axis else None,
                    cyclic_implementation_sha256=CYCLIC_IMPLEMENTATION_SHA256 if wrap_axis else None,
                    view_sampler_recipe=view_recipe,
                    canvas_transform=_plain(metadata.get("canvas_transform", {})),
                    resource_admission_identity=_plain(metadata.get("sam_resource_profile", {}).get("effective_budgets", {})),
                    upstream_lineage=_plain(upstream_lineage), spacing_zyx=list(spacing_zyx),
                    planner_limits=_plain(planner_limits if planner_limits is not None else SamPlanningLimits()),
                    view_name=str(getattr(view, "name", view)),
                    physical_view=str(getattr(view, "physical_view_name", "")),
                    angle_deg=float(getattr(view, "tta_angle_deg", 0.0) or 0.0),
                    augmentation_pass=int(getattr(view, "augmentation_pass", metadata.get("augmentation_pass", 0)) or 0),
                    augmentation_id=str(getattr(view, "tta_aug_id", "") or ""),
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
    cpu_wave_admission: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}), compare=False, repr=False)

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
        """Drain bounded cohorts; single workers finish each exact crop queue."""
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
            queues = {}
            for index in indices:
                queues.setdefault(self.tracker_jobs[index].tile.crop_bbox_yx, []).append(index)
            if int(worker_count)>1:
                indices = [queue[wave] for wave in range(max((len(q) for q in queues.values()), default=0))
                           for queue in queues.values() if wave < len(queue)]
            else:
                # Opposite endpoint sessions remain independent. Only their
                # immutable visual crop/frame products can be reused here.
                indices = [index for queue in queues.values() for index in queue]
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
    resource_profile: object = None,
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
    resource_details = None
    if resource_profile is not None:
        from .sam_resources import validate_live_sam_resource_profile
        resource_details = validate_live_sam_resource_profile(resource_profile)
        metadata["sam_resource_profile"] = resource_details
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
        limits=planner_limits, planning_pass_index=int(pass_index), wrap_axis=bool(wrap_axis), lazy_contracts=True)
    if resource_profile is not None and getattr(plan, "contract_lease_budget", None) is not None:
        plan.contract_lease_budget.bind_live_resource_profile(resource_profile)
    planning_seconds = time.perf_counter() - planning_started
    runs = tuple(run for run in plan.runs if int(run.pass_index) == int(pass_index))
    if runs:
        # These are runtime/measurement capacity checks, not quality selection.
        # Reject a request before rendering/model admission instead of silently
        # shortening a declared endpoint session or omitting a hypothesis.
        from .lta_sam import SamInterpolationSessionPlan
        for run in runs:
            try:
                SamInterpolationSessionPlan(sequence_id=str(run.run_id), session_index=0,
                    frame_start=min(run.expected_frames), frame_stop=max(run.expected_frames) + 1)
            except (TypeError, ValueError) as error:
                raise SamInterpolationInfrastructureError(
                    f"SAM planned run {run.run_id} is outside the bounded tracker session contract: {error}") from error
        from .sam_policy import resolve_sam_bridge_policy, _selection_resources
        source_policy = policy or {}
        if "kind" in source_policy and "mode" not in source_policy:
            source_policy = {"sam_bridge_policy": source_policy}
        topology_policy = resolve_sam_bridge_policy(source_policy, generation_mode=mode)
        topology_operative, _ = _selection_resources(source_policy, topology_policy, resource_profile)
        topology_limit = int(topology_operative['max_group_bytes'])
        tracked_group_ids = {str(run.group_id) for run in runs}
        for group in plan.groups:
            if str(group.group_id) not in tracked_group_ids:
                continue
            y0, x0, y1, x1 = group.context_bbox_yx
            topology_shape = (len(group.frame_indices), y1-y0, x1-x0)
            if topology_policy.get('branch_aware_selection', False):
                from .sam_branch_selection import branch_workspace_bytes
                topology_bytes = branch_workspace_bytes(topology_shape)
            else:
                topology_bytes = len(group.frame_indices)*(y1-y0)*(x1-x0)*16
            if topology_bytes > topology_limit:
                raise SamInterpolationInfrastructureError(
                    f"SAM group {group.group_id} topology exceeds its declared memory budget before tracker admission")
        from .sam_resources import cpu_session_bytes
        cpu_budget = 2 * 1024**3 if resource_details is None else int(resource_details["assigned_session_cpu_bytes"])
        groups_by_id = {group.group_id: group for group in plan.groups}
        if mode == "whole":
            for run in runs:
                y0, x0, y1, x1 = groups_by_id[run.group_id].context_bbox_yx
                estimate = cpu_session_bytes(len(run.expected_frames), (y1 - y0) * (x1 - x0))
                if estimate["estimated_peak_bytes"] > cpu_budget:
                    raise SamInterpolationInfrastructureError(
                        f"SAM run {run.run_id} known CPU session buffers require {estimate['estimated_peak_bytes']} "
                        f"bytes; admitted budget is {cpu_budget}; full interval was not started or truncated")
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
        from .sam_resources import cpu_session_bytes
        cpu_budget = 2 * 1024**3 if resource_details is None else int(resource_details["assigned_session_cpu_bytes"])
        for job in tracker_jobs:
            y0, x0, y1, x1 = job.tile.crop_bbox_yx
            estimate = cpu_session_bytes(len(job.original_run.expected_frames), (y1-y0)*(x1-x0))
            if estimate["estimated_peak_bytes"] > cpu_budget:
                raise SamInterpolationInfrastructureError(
                    f"SAM tile run {job.run_id} known CPU session buffers require {estimate['estimated_peak_bytes']} "
                    f"bytes; admitted budget is {cpu_budget}; full interval was not started or truncated")
        needed_frames = tuple(sorted({int(frame) for job in tracker_jobs for frame in job.original_run.expected_frames}))
        demand = {}
        for job in tracker_jobs:
            crop = job.tile.crop_bbox_yx
            for frame in job.original_run.expected_frames:
                old = demand.get(int(frame))
                demand[int(frame)] = crop if old is None else (
                    min(old[0], crop[0]), min(old[1], crop[1]), max(old[2], crop[2]), max(old[3], crop[3]))
        frame_crop_bounds = MappingProxyType(demand)
    wave_details = {}
    if resource_details is not None:
        from .sam_resources import cpu_session_bytes, cpu_wave_admission
        group_lookup = {str(group.group_id): group for group in plan.groups}
        max_session = max_raw = 0
        for work in (tracker_jobs if mode == 'tiled' else runs):
            run = work.original_run if mode == 'tiled' else work
            bbox = work.tile.crop_bbox_yx if mode == 'tiled' else group_lookup[str(run.group_id)].context_bbox_yx
            pixels = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            count = len(run.expected_frames)
            max_session = max(max_session, cpu_session_bytes(count, pixels)['estimated_peak_bytes'])
            max_raw = max(max_raw, count*pixels)
        try:
            wave_details = cpu_wave_admission(max_session, max_raw,
                resource_details['assigned_cpu_wave_bytes'], resource_details['worker_count'])
        except RuntimeError as error:
            raise SamInterpolationInfrastructureError(str(error)) from error
    return SamPreparedInterpolationPass(plan, runs, tuple(observations.shape),
        _buffer_identity(observations), snapshot, settings_sha256, planning_seconds,
        snapshot_seconds, needed_frames, frame_crop_bounds,
        canonical_buffer_identity=(None if canonical_labels is None else
                                   _buffer_identity(np.asarray(canonical_labels))),
        crop_mode=mode, tracker_jobs=tracker_jobs, tile_inventory=tile_inventory,
        tiling_sha256=tiling_sha256, tiled_assembly_bytes=assembly_bytes,
        cpu_wave_admission=MappingProxyType(wave_details))


def _tracker_requests(runs: tuple[object, ...], groups: Mapping[str, object],
                      observations: Mapping[str, object], cancel_event: object, resource_profile: object = None):
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
            crop_xyxy=(x0, y0, x1, y1),
            **({"resource_profile": resource_profile} if resource_profile is not None else {}))
        del seed_mask


def _tiled_tracker_requests(jobs, groups, observations, cancel_event, resource_profile=None):
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
                crop_mode="tiled", rule="1008/128/midpoint-v1"),
            **({"resource_profile": resource_profile} if resource_profile is not None else {}))
        del seed


def _family_dispatch_balance(runs, flat_execution_order, admitted_slots):
    """Compare scheduling imbalance using sealed frame counts, never image work.

    This deterministic proxy ignores cache savings and predicts no runtime. It
    protects the automatic default when pinning families loses substantial job
    parallelism; explicit scheduling choices remain caller-owned.
    """
    slots = int(admitted_slots)
    if slots < 1:
        raise ValueError("SAM scheduling balance requires an admitted worker")
    frame_work = tuple(len(run.expected_frames) for run in runs)
    family_work = {}
    for run, frames in zip(runs, frame_work):
        identity = str(run.group_id)
        family_work[identity] = family_work.get(identity, 0) + frames

    def span(weights):
        lanes = [(0, lane) for lane in range(slots)]
        heapq.heapify(lanes)
        for weight in weights:
            previous, lane = heapq.heappop(lanes)
            heapq.heappush(lanes, (previous + weight, lane))
        return max(value for value, lane in lanes)

    flat_span = span(frame_work[index] for index in flat_execution_order)
    family_span = span(family_work.values())
    return dict(schema="xta.sam_family_dispatch_balance/1", basis="expected_frame_count",
        admitted_slots=slots, logical_run_count=len(runs), family_count=len(family_work),
        flat_job_makespan_frames=flat_span, fifo_family_makespan_frames=family_span,
        fifo_to_flat_ratio=(family_span / flat_span if flat_span else None),
        fallback_flat=family_span * 4 > flat_span * 5, walltime_prediction=False)


def _iterate_tracker_results(runtime: object, requests: object, cache_ref: object, cpu_wave=None):
    if hasattr(runtime, "iter_results"):
        options = {} if not cpu_wave else dict(max_in_flight=int(cpu_wave['max_in_flight']),
            defer_refill_until_consumed=bool(cpu_wave['defer_refill_until_consumed']))
        if options and getattr(runtime, 'device_ids', None):
            options['max_in_flight'] = min(options['max_in_flight'], len(runtime.device_ids))
        yield from runtime.iter_results(requests, source_cache_ref=cache_ref, **options)
    else:
        for index, request in enumerate(requests):
            yield index, runtime.run(**request)


def _iterate_tiled_tracker_results(runtime, prepared, worker_count, groups, observations, cache_ref, cancel_event, resource_profile=None):
    offset = 0
    for batch in prepared.execution_batches(worker_count):
        jobs = tuple(prepared.tracker_jobs[index] for index in batch)
        stream = _iterate_tracker_results(runtime,
            _tiled_tracker_requests(jobs, groups, observations, cancel_event, resource_profile), cache_ref,
            prepared.cpu_wave_admission)
        try:
            for index, result in stream:
                valid = not isinstance(index, bool) and isinstance(index, (int, np.integer)) and 0<=int(index)<len(batch)
                yield offset+int(index) if valid else -1, result
                del result
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
            {"observation_id": str(identifier),
             "original_observation_id": str(getattr(observation_by_id[str(identifier)], "original_observation_id", "") or identifier),
             "native_frame_index": int(getattr(observation_by_id[str(identifier)], "native_frame_index", None)
                if getattr(observation_by_id[str(identifier)], "native_frame_index", None) is not None else observation_by_id[str(identifier)].frame_index),
             "lineage": _plain(observation_by_id[str(identifier)].lineage)}
            for identifier in tuple(run.seed_ids) + tuple(run.held_out_ids)],
        "upstream": _plain(upstream_lineage)}
    raw_masks = {int(frame): np.asarray(mask) for frame, mask in result.frames.items()}
    # The writer already owns the exact staged per-edge write contracts. It
    # clips one raw plane at a time using run.edge_ids, without retaining or
    # rematerializing a family's dense planning masks throughout inference.
    if availability_masks is None:
        writer.add_run(descriptor, raw_masks)
    else:
        descriptor["sam_crop_mode"] = "tiled"
        writer.add_run(descriptor, raw_masks, availability_masks=availability_masks)
    if not bool(descriptor["structurally_valid"]):
        raise SamInterpolationInfrastructureError("SAM tracker produced structurally invalid expected object evidence")
    return descriptor


def _retry_work(prepared):
    """Count complete SDK jobs, including independent tiled halo sessions."""
    grouped = {}
    groups = {str(group.group_id): group for group in prepared.groups}
    for item in (prepared.tracker_jobs if prepared.crop_mode == 'tiled' else prepared.runs):
        run = item.original_run if prepared.crop_mode == 'tiled' else item
        box = item.tile.crop_bbox_yx if prepared.crop_mode == 'tiled' else groups[str(run.group_id)].context_bbox_yx
        frames = len(run.expected_frames)
        work, count = grouped.get(str(run.group_id), (0, 0))
        grouped[str(run.group_id)] = (work + frames*(box[2]-box[0])*(box[3]-box[1]), count + frames)
    return grouped


def _retry_child_crop_contacts(bundle, reader, group_id, canvas_shape_yx):
    from .sam_crop_retry import raw_child_crop_boundary_contacts, summarize_child_crop_contacts
    group = bundle.groups[str(group_id)]
    def records():
        for run_id, run in bundle.runs.items():
            if str(run['group_id']) != str(group_id):
                continue
            for tile in run.get('tile_evidence', ()):
                if not tile.get('attempted'):
                    continue
                for frame in run['expected_frames']:
                    raw = reader.tile_raw_mask(run_id, str(tile['tile_id']), int(frame))
                    yield raw_child_crop_boundary_contacts(raw, tile['crop_bbox_yx'],
                        group['context_bbox_yx'], canvas_shape_yx)
                    del raw
    return summarize_child_crop_contacts(records())


def _retry_cpu_wave_within_peak(serial_wave, *, approved_peak_bytes, image_bytes,
                                worker_count, persistent_bytes=0):
    """Reuse already approved phase credit without changing retry admission.

    Rendering/selection scratch can dominate the serial retry's peak. Independent
    endpoint sessions may share that credit while those CPU phases are inactive.
    Full contract, writer and tiled assembly owners remain separately charged.
    The original serial estimate still decides crop eligibility; a larger GPU
    wave must fit both its owned CPU credit and that unchanged approved peak.
    """
    from .sam_resources import cpu_wave_admission
    original = dict(serial_wave)
    workers = max(1, int(worker_count))
    if workers == 1 or original['defer_refill_until_consumed']:
        return original
    wave_bytes = min(int(original['assigned_cpu_wave_bytes']),
        int(approved_peak_bytes)-int(image_bytes)-int(persistent_bytes))
    if wave_bytes < int(original['peak_cpu_wave_estimate_bytes']):
        return original
    admitted = cpu_wave_admission(int(original['maximum_session_cpu_estimate_bytes']),
        int(original['maximum_raw_mask_bytes']), wave_bytes, workers)
    if (int(admitted['max_in_flight']) <= int(original['max_in_flight'])
            or int(admitted['peak_cpu_wave_estimate_bytes'])+int(image_bytes)+int(persistent_bytes)
                > int(approved_peak_bytes)):
        return original
    return dict(admitted, retry_approved_peak_bytes=int(approved_peak_bytes),
        retry_image_bytes=int(image_bytes), retry_persistent_bytes=int(persistent_bytes))


def _prepare_sam_group_retry(prepared, group, crop, *, retry_policy, policy,
        resource_profile, worker_count):
    """Prepare only the same original family under a fresh enlarged context."""
    from .sam_bridge_planning import expand_sam_bridge_group_context
    from .sam_policy import resolve_sam_bridge_policy, _selection_resources
    from .sam_resources import cpu_session_bytes, cpu_wave_admission, validate_live_sam_resource_profile
    from .workspace import _env_int
    lineage = dict(schema='xta.sam_crop_retry_attempt/1', original_group_id=str(group.group_id),
        original_context_bbox_yx=list(group.context_bbox_yx), retry_index=1,
        original_observation_snapshot_sha256=prepared.observation_snapshot_sha256)
    enlarged = expand_sam_bridge_group_context(group, crop,
        max_crop_pixels=retry_policy.max_crop_pixels, retry_lineage=lineage)
    original_runs = tuple(run for run in prepared.runs if str(run.group_id) == str(group.group_id))
    runs = tuple(replace(run, group_id=enlarged.group_id,
        run_id=str(run.run_id) + '__retry1_' + hashlib.sha256(enlarged.group_id.encode()).hexdigest()[:12])
        for run in original_runs)
    plan = replace(prepared.plan, groups=(enlarged,), runs=runs,
        planning_fingerprint=hashlib.sha256((prepared.settings_sha256 + enlarged.group_id).encode()).hexdigest())
    jobs, inventory, tiling_sha, assembly_bytes = (), MappingProxyType({}), '', 0
    demand = plan.frame_crop_bounds
    if prepared.crop_mode == 'tiled':
        from .sam_crop_tiling import prepare_tiled_jobs
        jobs, inventory, tiling_sha, assembly_bytes = prepare_tiled_jobs(runs,
            {enlarged.group_id: enlarged}, plan.by_id)
        tiled_demand = {}
        for job in jobs:
            box = job.tile.crop_bbox_yx
            for frame in job.original_run.expected_frames:
                old = tiled_demand.get(int(frame), box)
                tiled_demand[int(frame)] = (min(old[0], box[0]), min(old[1], box[1]),
                    max(old[2], box[2]), max(old[3], box[3]))
        demand = MappingProxyType(tiled_demand)
    source_policy = policy or {}
    if 'kind' in source_policy and 'mode' not in source_policy:
        source_policy = {'sam_bridge_policy': source_policy}
    bridge_policy = resolve_sam_bridge_policy(source_policy, generation_mode=prepared.crop_mode)
    operative, _ = _selection_resources(source_policy, bridge_policy, resource_profile)
    shape = (len(enlarged.frame_indices), crop[2]-crop[0], crop[3]-crop[1])
    if bridge_policy.get('branch_aware_selection', False):
        from .sam_branch_selection import branch_workspace_bytes
        topology_bytes = branch_workspace_bytes(shape)
    else:
        topology_bytes = int(np.prod(shape))*16
    if topology_bytes > int(operative['max_group_bytes']):
        raise MemoryError('Enlarged SAM family exceeds its current topology admission')
    profile = (validate_live_sam_resource_profile(resource_profile)
        if resource_profile is not None else None)
    session_cap = 2*1024**3 if profile is None else int(profile['assigned_session_cpu_bytes'])
    owned_wave = 2*1024**3 if profile is None else int(profile['assigned_cpu_wave_bytes'])
    max_session = max_raw = 0
    for item in (jobs if prepared.crop_mode == 'tiled' else runs):
        run = item.original_run if prepared.crop_mode == 'tiled' else item
        box = item.tile.crop_bbox_yx if prepared.crop_mode == 'tiled' else crop
        pixels = (box[2]-box[0])*(box[3]-box[1])
        peak = cpu_session_bytes(len(run.expected_frames), pixels)['estimated_peak_bytes']
        if peak > session_cap:
            raise MemoryError('Enlarged SAM full interval exceeds its current CPU session admission')
        max_session, max_raw = max(max_session, peak), max(max_raw, pixels*len(run.expected_frames))
    gray_bytes = sum((box[2]-box[0])*(box[3]-box[1]) for box in demand.values())
    cache_cap = max(1, _env_int('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', 1024**3))
    if gray_bytes > cache_cap:
        raise MemoryError('Enlarged SAM image demand exceeds its immutable cache admission')
    render_reserve = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
    # Preserve the serial estimate used to decide whether this crop is eligible.
    # A larger worker wave may spend only credit already inside that peak.
    wave_budget = min(owned_wave, max(1, retry_policy.max_retry_memory_bytes-gray_bytes))
    wave = cpu_wave_admission(max_session, max_raw, wave_budget, 1)
    peak = max(int(enlarged.crop_contract['charged_contract_bytes']) + 32*1024**2,
        topology_bytes + 32*1024**2, int(wave['peak_cpu_wave_estimate_bytes']) + gray_bytes,
        2*gray_bytes + render_reserve, int(assembly_bytes) + gray_bytes)
    wave = _retry_cpu_wave_within_peak(wave, approved_peak_bytes=peak,
        image_bytes=gray_bytes, worker_count=worker_count,
        persistent_bytes=int(assembly_bytes)+int(enlarged.crop_contract['charged_contract_bytes'])+32*1024**2)
    retry = replace(prepared, plan=plan, runs=runs, needed_frames=plan.needed_frames,
        frame_crop_bounds=demand, tracker_jobs=jobs, tile_inventory=inventory,
        tiling_sha256=tiling_sha, tiled_assembly_bytes=assembly_bytes,
        cpu_wave_admission=MappingProxyType(wave),
        settings_sha256=hashlib.sha256((prepared.settings_sha256+enlarged.group_id).encode()).hexdigest())
    return retry, peak, {str(new.run_id): str(old.run_id) for old, new in zip(original_runs, runs)}


def _regenerate_sam_group_retry(prepared, *, destination, metadata, image_provider,
        runtime, resource_profile, upstream_lineage, min_radius, cancel_event, original_run_ids):
    """Track only the enlarged family, retaining its own immutable raw attempt."""
    from .sam_evidence import SamEvidenceWriter
    from .sam_crop_tiling import TiledRunAssembly, tile_descriptor
    groups = {str(group.group_id): group for group in prepared.groups}
    by_id = prepared.plan.by_id
    cache = getattr(image_provider, 'cache_ref', image_provider)
    writer = SamEvidenceWriter(destination / 'evidence', metadata)
    assemblies = {}
    stream = None
    completed = set()
    try:
        for group in prepared.groups:
            _write_group(writer, group, by_id, min_radius)
        tiled = prepared.crop_mode == 'tiled'
        work = prepared.tracker_jobs if tiled else prepared.runs
        if tiled:
            for index, tiles in prepared.tile_inventory.items():
                for tile in tiles:
                    if not tile.attempted:
                        writer.add_run_tile(prepared.runs[index].run_id,
                            tile_descriptor(prepared.runs[index], tile), {})
        worker_count = min(len(getattr(runtime, 'device_ids', ())) or 1,
            int(prepared.cpu_wave_admission.get('max_in_flight', 1)))
        order = prepared.execution_order(worker_count)
        stream = (_iterate_tiled_tracker_results(runtime, prepared, worker_count, groups, by_id, cache,
            cancel_event, resource_profile) if tiled else _iterate_tracker_results(runtime,
            _tracker_requests(tuple(prepared.runs[i] for i in order), groups, by_id,
                cancel_event, resource_profile), cache, prepared.cpu_wave_admission))
        for index, result in stream:
            try:
                if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(order):
                    raise SamInterpolationInfrastructureError('SAM retry returned invalid job ownership')
                work_index = order[int(index)]
                if work_index in completed:
                    raise SamInterpolationInfrastructureError('SAM retry returned duplicate job ownership')
                item = work[work_index]
                run = item.original_run if tiled else item
                group = groups[str(run.group_id)]
                expected_id = item.run_id if tiled else run.run_id
                declared = getattr(result, 'receipt', {}).get('run_id')
                if declared is not None and str(declared) != str(expected_id):
                    raise SamInterpolationInfrastructureError('SAM retry changed its job identity')
                if tiled:
                    for identity_key in ('sam_model', 'sam_runtime'):
                        actual = dict(result.receipt or {}).get(identity_key)
                        if actual is not None:
                            actual = _plain(actual)
                            if identity_key in metadata and metadata[identity_key] != actual:
                                raise SamInterpolationInfrastructureError('SAM retry tile changed its model/runtime identity')
                            metadata[identity_key] = actual
                            writer.scope[identity_key] = actual
                    parent = item.original_run_index
                    if parent not in assemblies:
                        assemblies[parent] = TiledRunAssembly(run, group, prepared.tile_inventory[parent],
                            destination / 'tile_assemblies' / str(run.run_id))
                    assembly = assemblies[parent]
                    child, masks = assembly.consume(item, result)
                    writer.add_run_tile(run.run_id, child, masks)
                    if not child['structurally_valid']:
                        raise SamInterpolationInfrastructureError('SAM retry returned invalid tiled raw evidence')
                    if assembly.ready:
                        full = assembly.result()
                        for identity_key in ('sam_model', 'sam_runtime'):
                            if identity_key in metadata:
                                full.receipt[identity_key] = metadata[identity_key]
                        descriptor = _store_generated_parent_run(writer, run, full, group, by_id,
                            metadata, upstream_lineage, assembly.availability())
                        descriptor['crop_retry_of_run_id'] = original_run_ids[str(run.run_id)]
                        writer.runs[str(run.run_id)]['crop_retry_of_run_id'] = original_run_ids[str(run.run_id)]
                        del full
                        assembly.close()
                        del assemblies[parent]
                else:
                    _store_generated_parent_run(writer, run, result, group, by_id, metadata, upstream_lineage)
                    writer.runs[str(run.run_id)]['crop_retry_of_run_id'] = original_run_ids[str(run.run_id)]
                completed.add(work_index)
            finally:
                if hasattr(runtime, 'release_result'):
                    runtime.release_result(result)
                del result
        if len(completed) != len(work) or assemblies:
            raise SamInterpolationInfrastructureError('SAM retry omitted its complete original interval')
        return writer.commit(complete=True)
    except BaseException as error:
        try:
            writer.commit(complete=False)
        except BaseException:
            pass
        _failure_receipt(destination, error, 'crop_retry_generation', len(completed))
        raise
    finally:
        for assembly in assemblies.values():
            assembly.close()
        if stream is not None:
            stream.close()


def _apply_sam_crop_retries(bundle, prepared, *, destination, metadata, runtime,
        retry_policy, retry_image_provider, policy, resource_profile, min_radius,
        upstream_lineage, cancel_event):
    """Choose one full attempt per family, then preserve global quality gates."""
    if retry_policy is None or not retry_policy.enabled or not prepared.runs:
        return bundle, destination, None
    from .sam_crop_retry import (SamCropRetryController, raw_crop_boundary_contacts, merge_crop_contacts)
    from .sam_evidence import SamEvidenceWriter, fingerprint
    from .sam_resources import validate_live_sam_resource_profile
    work = _retry_work(prepared)
    controller = SamCropRetryController(retry_policy,
        baseline_pixel_frames=sum(value[0] for value in work.values()),
        baseline_tracker_frames=sum(value[1] for value in work.values()),
        largest_original_group_pixel_frames=max(value[0] for value in work.values()),
        largest_original_group_tracker_frames=max(value[1] for value in work.values()))
    profile = validate_live_sam_resource_profile(resource_profile) if resource_profile is not None else None
    available = 2*1024**3 if profile is None else int(profile['assigned_cpu_wave_bytes'])
    replacements = {}
    attempts = []
    preflight_refusals = {}
    tiled_child_contacts = {}
    with bundle.reader(max_cache_bytes=0) as reader:
        for group in prepared.groups:
            group_id = str(group.group_id)
            if group_id not in work:
                continue
            _check_cancelled(cancel_event)
            original_runs = tuple(run for run in prepared.runs if str(run.group_id) == group_id)
            seed_identity = fingerprint(dict(snapshot=prepared.observation_snapshot_sha256,
                seeds=[(str(run.run_id), tuple(run.seed_ids)) for run in original_runs]))
            interval_identity = fingerprint([(str(run.run_id), tuple(run.expected_frames), int(run.direction))
                for run in original_runs])
            contacts = []
            for run in original_runs:
                for frame in run.expected_frames:
                    if prepared.crop_mode == 'tiled':
                        raw = reader.halo_union_mask(str(run.run_id), int(frame))
                        contacts.append(raw_crop_boundary_contacts(raw, group.context_bbox_yx,
                            prepared.native_shape[1:]))
                        del raw
                    else:
                        contacts.append(reader.raw_crop_boundary_contacts(str(run.run_id), int(frame),
                            crop_bbox_yx=group.context_bbox_yx, canvas_shape_yx=prepared.native_shape[1:]))
            if prepared.crop_mode == 'tiled':
                tiled_child_contacts[group_id] = {'initial': _retry_child_crop_contacts(
                    bundle, reader, group_id, prepared.native_shape[1:])}
            contacts = merge_crop_contacts(contacts)
            candidates = {}
            def candidate(box):
                if box not in candidates:
                    candidates[box] = _prepare_sam_group_retry(prepared, group, box,
                        retry_policy=retry_policy, policy=policy, resource_profile=resource_profile,
                        worker_count=len(getattr(runtime, 'device_ids', ())) or 1)
                return candidates[box]
            def memory(box):
                try:
                    return candidate(box)[1]
                except Exception as error:
                    preflight_refusals[group_id] = str(error)
                    return max(available, retry_policy.max_retry_memory_bytes) + 1
            def pixel_work(box):
                try:
                    return sum(value[0] for value in _retry_work(candidate(box)[0]).values())
                except Exception as error:
                    preflight_refusals[group_id] = str(error)
                    return retry_policy.max_extra_pixel_frames + 1
            def frame_work(box):
                try:
                    return sum(value[1] for value in _retry_work(candidate(box)[0]).values())
                except Exception:
                    return 513
            decision = controller.reserve_retry(group_id, crop_bbox_yx=group.context_bbox_yx,
                canvas_shape_yx=prepared.native_shape[1:], frame_count=sum(len(run.expected_frames) for run in original_runs),
                seed_identity=seed_identity, interval_identity=interval_identity, contacts=contacts,
                available_memory_bytes=available, memory_estimator=memory,
                work_estimator=pixel_work if prepared.crop_mode == 'tiled' else None,
                tracker_frame_estimator=frame_work if prepared.crop_mode == 'tiled' else None)
            if not decision.retry:
                continue
            retry, _, run_ids = candidate(decision.crop_bbox_yx)
            controller.verify_retry_identity(decision, seed_identity=seed_identity, interval_identity=interval_identity)
            retry_destination = destination / 'crop_retries' / hashlib.sha256(group_id.encode()).hexdigest()[:20]
            retry_destination.mkdir(parents=True, exist_ok=True)
            attempt_metadata = dict(metadata, sam_crop_retry_attempt=decision.record,
                sam_crop_retry_policy=retry_policy.to_dict(),
                retry_tiling_plan_sha256=retry.tiling_sha256)
            try:
                if not callable(retry_image_provider):
                    raise ValueError('Adaptive SAM crop cannot render without a fresh image-provider factory')
                provider = retry_image_provider(retry)
                cache = getattr(provider, 'cache_ref', provider)
                if cache is not None:
                    if tuple(cache.shape) != tuple(retry.plan.virtual_shape_tyx or retry.native_shape):
                        raise ValueError('SAM retry image descriptor changed the original canvas/frame geometry')
                    coverage = {int(record[0]): tuple(map(int, record[1:5]))
                        for record in getattr(cache, 'frame_crops', ())}
                    if coverage:
                        for frame, box in retry.frame_crop_bounds.items():
                            present = coverage.get(int(frame))
                            if present is None or present[0] > box[0] or present[1] > box[1] or present[2] < box[2] or present[3] < box[3]:
                                raise ValueError('SAM retry image descriptor does not cover the enlarged context')
                    if hasattr(cache, 'revalidate'):
                        cache.revalidate()
                    image_identity = str(getattr(cache, 'identity_sha256', ''))
                    pinned_image_identity = str(metadata.get('image_snapshot_sha256', ''))
                    if image_identity and pinned_image_identity and image_identity != pinned_image_identity:
                        raise ValueError('SAM retry image descriptor changed its pinned canonical source-image identity')
                    attempt_metadata['retry_image_snapshot_sha256'] = image_identity
                    attempt_metadata['retry_image_frame_crops'] = _plain(getattr(cache, 'frame_crops', ()))
                attempt = _regenerate_sam_group_retry(retry, destination=retry_destination,
                    metadata=attempt_metadata, image_provider=provider, runtime=runtime,
                    resource_profile=resource_profile, upstream_lineage=upstream_lineage,
                    min_radius=min_radius, cancel_event=cancel_event, original_run_ids=run_ids)
                retry_contacts = []
                with attempt.reader(max_cache_bytes=0) as attempt_reader:
                    for retry_run in retry.runs:
                        for frame in retry_run.expected_frames:
                            if prepared.crop_mode == 'tiled':
                                raw = attempt_reader.halo_union_mask(str(retry_run.run_id), int(frame))
                                retry_contacts.append(raw_crop_boundary_contacts(raw, decision.crop_bbox_yx,
                                    prepared.native_shape[1:]))
                                del raw
                            else:
                                retry_contacts.append(attempt_reader.raw_crop_boundary_contacts(
                                    str(retry_run.run_id), int(frame), crop_bbox_yx=decision.crop_bbox_yx,
                                    canvas_shape_yx=prepared.native_shape[1:]))
                    if prepared.crop_mode == 'tiled':
                        tiled_child_contacts[group_id]['retry'] = _retry_child_crop_contacts(
                            attempt, attempt_reader, retry.groups[0].group_id, prepared.native_shape[1:])
                retry_contacts = merge_crop_contacts(retry_contacts)
                controller.complete_retry(decision, status='succeeded',
                    detail={'evidence_path': str(attempt.directory), 'chosen_group_ids': list(attempt.groups),
                        'outer_context_contacts': retry_contacts,
                        'extent_remains_censored': any(retry_contacts['internal_contacts'].values()),
                        'maximum_additional_attempts': 0, 'coverage_proof': False})
                replacements[group_id] = attempt
                attempts.append(attempt)
            except BaseException as error:
                controller.complete_retry(decision, status='cancelled' if getattr(cancel_event, 'is_set', lambda: False)()
                    else 'failed', detail={'error': str(error)})
                (destination/'crop_retry.json').write_text(json.dumps(controller.receipt(), indent=2), encoding='utf-8')
                runtime_cancel = getattr(runtime, '_cancel', None)
                cancelled = getattr(cancel_event, 'is_set', lambda: False)() or (
                    runtime_cancel is not None and getattr(runtime_cancel, 'is_set', lambda: False)() is True)
                if cancelled or isinstance(error, (KeyboardInterrupt, SystemExit)) or getattr(runtime, '_closed', False) is True:
                    raise
                # The complete initial attempt remains authoritative for this
                # optional retry failure. Drop closed traceback-frame aliases
                # before another group or global selection borrows workspace.
                import traceback
                traceback.clear_frames(error.__traceback__)
                continue
    receipt = controller.receipt()
    receipt['preflight_refusals'] = preflight_refusals
    receipt['tiled_child_crop_contacts'] = tiled_child_contacts
    receipt['initial_evidence_path'] = str(bundle.directory)
    receipt['initial_evidence_fingerprint'] = bundle.evidence_fingerprint
    (destination/'crop_retry.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    if not replacements:
        return bundle, destination, receipt
    final_destination = destination/'adaptive_final'
    final_destination.mkdir(parents=True, exist_ok=True)
    final_metadata = dict(metadata, sam_crop_retry=receipt,
        chosen_attempt_basis='one_complete_raw_attempt_per_original_family_global_reselection')
    final_writer = SamEvidenceWriter(final_destination/'evidence', final_metadata)
    try:
        with final_writer.import_transaction(bundle):
            for group_id in bundle.groups:
                if group_id not in replacements:
                    final_writer.import_group(bundle, group_id)
            for run_id, run in bundle.runs.items():
                if str(run['group_id']) not in replacements:
                    final_writer.import_run(bundle, run_id)
        for attempt in attempts:
            with final_writer.import_transaction(attempt):
                for group_id in attempt.groups:
                    final_writer.import_group(attempt, group_id)
                for run_id in attempt.runs:
                    final_writer.import_run(attempt, run_id)
        final_bundle = final_writer.commit(complete=True)
    except BaseException:
        final_writer.abort()
        raise
    receipt['final_evidence_path'] = str(final_bundle.directory)
    receipt['final_evidence_fingerprint'] = final_bundle.evidence_fingerprint
    (destination/'crop_retry.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return final_bundle, final_destination, receipt


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
    from .sam_view_geometry import validate_sam_view_geometry
    validate_sam_view_geometry(view, wrap_axis=wrap_axis, scope=scope)


def _frame_masks(masks: object, frames: tuple[int, ...]) -> dict[int, np.ndarray]:
    if isinstance(masks, Mapping):
        return {int(frame): np.asarray(mask, dtype=bool) for frame, mask in masks.items()}
    array = np.asarray(masks, dtype=bool)
    if array.ndim != 3 or len(array) != len(frames):
        raise ValueError("SAM plan masks must correspond exactly to group frame_indices")
    return {int(frame): array[index] for index, frame in enumerate(frames)}


class _StreamingGroupMasks(Mapping):
    """Endpoint crop planes are transient; contract planes remain sealed views."""
    def __init__(self):
        self.entries = {}

    def __getitem__(self, name):
        value = self.entries[name]
        return value() if callable(value) else value

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __setitem__(self, name, value):
        self.entries[name] = value


def _write_group(writer: object, group: object, observations: Mapping[str, object],
                 min_radius: float) -> None:
    materialize=getattr(group,"materialize_contracts",None)
    if callable(materialize):
        # One sealed family is streamed to indexed evidence, then its dense
        # contracts leave scope before the next family/model work is admitted.
        with materialize() as concrete:
            _write_materialized_group(writer,concrete,observations,min_radius)
    else:
        # Literal historical/research dataclasses retain their existing API.
        _write_materialized_group(writer,group,observations,min_radius)


def _write_materialized_group(writer: object, group: object, observations: Mapping[str, object],
                              min_radius: float) -> None:
    frames = tuple(int(frame) for frame in group.frame_indices)
    bbox = tuple(int(value) for value in group.context_bbox_yx)
    crop_metadata = ({"crop_contract": _plain(group.crop_contract)} if getattr(group,"crop_contract",None) else {})
    if getattr(group, "frame_addressing", None):
        crop_metadata.update(frame_addressing=_plain(group.frame_addressing),
                             frame_addresses=_plain(group.frame_addresses),
                             native_shape_tyx=list(group.native_shape_tyx))
    if str(group.status) in {"unresolved", "incomplete", "invalid"}:
        # Capped groups intentionally have no allocated crop contracts. Persist
        # their bounds/reasons without allocating the very resource they exceed.
        writer.add_group({
            "group_id": str(group.group_id), "context_bbox_yx": bbox,
            "frame_indices": frames, "endpoints": [
                {"observation_id": str(observations[str(identifier)].observation_id),
                 "frame_index": int(observations[str(identifier)].frame_index),
                 "canonical_label": int(observations[str(identifier)].canonical_label),
                 "original_observation_id": str(getattr(observations[str(identifier)], "original_observation_id", "") or identifier),
                 "native_frame_index": int(getattr(observations[str(identifier)], "native_frame_index", None)
                                            if getattr(observations[str(identifier)], "native_frame_index", None) is not None else observations[str(identifier)].frame_index),
                 "mirror_u": bool(getattr(observations[str(identifier)], "mirror_u", False)),
                 "bbox_yx": list(observations[str(identifier)].bbox_yx),
                 "lineage": _plain(observations[str(identifier)].lineage)}
                for identifier in group.observation_ids],
            "edges": [_plain(edge) for edge in group.edges],
            "status": str(group.status), "reasons": list(group.reasons),
            "complete": False, "interpolation_min_radius": float(min_radius),
            "connectivity": 6, "endpoint_identity_basis": "slice_connected_components",
            **crop_metadata,
        }, {f"endpoint_local:{identifier}": observations[str(identifier)].mask_crop
            for identifier in group.observation_ids})
        return
    acceptance = _frame_masks(group.acceptance_masks, frames)
    known = _frame_masks(group.known_foreground_masks, frames) if hasattr(group, "known_foreground_masks") else {}
    evaluations = getattr(group, "branch_evaluation_masks", {})
    endpoints = []
    masks = _StreamingGroupMasks()
    for observation_id in group.observation_ids:
        observation = observations[str(observation_id)]
        endpoints.append({
            "observation_id": str(observation.observation_id),
            "frame_index": int(observation.frame_index),
            "canonical_label": int(observation.canonical_label),
            "original_observation_id": str(getattr(observation, "original_observation_id", "") or observation.observation_id),
            "native_frame_index": int(getattr(observation, "native_frame_index", None) if getattr(observation, "native_frame_index", None) is not None else observation.frame_index),
            "mirror_u": bool(getattr(observation, "mirror_u", False)),
            "lineage": _plain(observation.lineage),
        })
        masks[f"endpoint:{observation_id}"] = lambda observation=observation: observation.mask_in_crop(bbox)
        # Nonterminal continuation/walk-back observations still need portable
        # references. Only held-out terminal domains are scored by the policy.
        masks[f"evaluation:{observation_id}"] = np.asarray(
            evaluations.get(str(observation_id), acceptance[int(observation.frame_index)]), dtype=bool)
        if int(observation.frame_index) in known:
            permitted = getattr(group, "branch_permitted_masks", {}).get(str(observation_id))
            masks[f"permitted:{observation_id}"] = (
                np.asarray(permitted, dtype=bool) if permitted is not None else
                lambda observation=observation: known[int(observation.frame_index)] & ~observation.mask_in_crop(bbox))
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
        **crop_metadata,
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
        group = groups[str(run["group_id"])]
        addresses = None
        if group.get("frame_addressing"):
            from .sam_cyclic import (address_for_unfolded_index, transform_crop_between_frame_addresses,
                                     validate_cyclic_frame_addressing)
            addresses = validate_cyclic_frame_addressing(group["frame_addressing"], expected_frames=group["frame_indices"])
            native_shape = tuple(group["frame_addressing"]["native_shape_tyx"])
            if native_shape[1:] != tuple(shape_yx):
                raise SamInterpolationInfrastructureError("Cyclic native output shape differs from saved addresses")
        for stored_frame in run["expected_frames"]:
            stored_frame = int(stored_frame)
            address = addresses[stored_frame] if addresses is not None else None
            if int(address["native_index"] if address is not None else stored_frame) != int(frame):
                continue
            crop = effective_candidate_mask(bundle, str(run_id), stored_frame, receipt)
            if crop is None:
                continue
            crop = np.asarray(crop, dtype=bool)
            bbox = tuple(map(int, group["context_bbox_yx"]))
            if address is not None:
                target = address_for_unfolded_index(int(frame), native_shape[0],
                    period_degrees=group["frame_addressing"]["period_degrees"])
                crop, bbox = transform_crop_between_frame_addresses(crop, bbox, address, target, shape_yx[1])
            y0, x0, y1, x1 = bbox
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
            selection_view.pop('branch_selection', None)
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
            active_frames_set = set()
            for identifier in directional_run_ids:
                group = groups[str(runs[identifier]["group_id"])]
                addresses = None
                if group.get("frame_addressing"):
                    from .sam_cyclic import validate_cyclic_frame_addressing
                    addresses = validate_cyclic_frame_addressing(group["frame_addressing"],
                        expected_frames=group["frame_indices"])
                branch = receipt.get('branch_selection')
                if branch is not None:
                    active_owner_frames = {int(frame) for edge_id in branch['selected_edge_ids_by_run'].get(identifier, ())
                        for frame in branch['edges'][edge_id]['owner_support'][identifier]}
                else:
                    active_owner_frames = {int(frame) for frame, key in runs[identifier]['candidate_mask_keys'].items()
                                           if int(bundle.records[key]['foreground']) > 0}
                for frame in sorted(active_owner_frames):
                    native_frame = int(addresses[int(frame)]["native_index"]) if addresses is not None else int(frame)
                    if not 0 <= native_frame < shape[0]:
                        raise SamInterpolationInfrastructureError("Selected SAM frame is outside its native publication canvas")
                    active_frames_set.add(native_frame)
            active_frames = sorted(active_frames_set)
            path = destination / f"sam_bridge_pass{pass_index:02d}_{direction}.cvol"
            writer = IncrementalRawBBoxMaskStoreWriter(
                shape=shape, store_dir=path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                desc=f"Selected SAM {direction} pass {pass_index}",
                extra_meta={**dict(metadata), "interpolation_backend": "sam",
                            "interpolation_direction": direction,
                            "interpolation_pass_index": int(pass_index),
                            "sam_policy_hash": str(receipt["policy_hash"]),
                            "sam_selection_identity": str(receipt.get("selection_identity", "")),
                            "sam_selection_resources": _plain(receipt.get("selection_resources", {})),
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
                store_meta.update(sam_selection_identity=str(receipt.get('selection_identity', '')),
                    sam_selection_resources=_plain(receipt.get('selection_resources', {})))
            except BaseException as error:
                writer.abort(error)
                raise
            components.append({
                "direction": direction, "path": str(path),
                "storage_format": INTERNAL_PACKED_CVOL_FORMAT,
                "voxel_count": int(store_meta.get("foreground_voxels", store_meta.get("added_voxels", 0))),
                "evidence_path": str(bundle.directory),
                "policy_hash": str(receipt["policy_hash"]), "metadata": store_meta,
                "sam_selection_identity": str(receipt.get("selection_identity", "")),
                "selection_resources": _plain(receipt.get("selection_resources", {})),
                "selection_receipt_path": str(selection_path),
            })
            directional_group_ids = sorted({str(runs[identifier]["group_id"]) for identifier in directional_run_ids})
            observation_roots = set()
            for run_id in directional_run_ids:
                endpoints = {str(endpoint["observation_id"]): endpoint for endpoint in
                             groups[str(runs[run_id]["group_id"])].get("endpoints", ())}
                roots = list(runs[run_id].get('seed_ids', ()))
                branch = receipt.get('branch_selection')
                if branch is None:
                    roots.extend(runs[run_id].get('held_out_ids', ()))
                else:
                    for edge_id in branch['selected_edge_ids_by_run'].get(run_id, ()):
                        roots.extend(branch['edges'][edge_id][key] for key in ('source_id', 'target_id'))
                for identifier in roots:
                        observation_roots.add(str(endpoints.get(str(identifier), {}).get("original_observation_id", identifier)))
            observation_roots = sorted(observation_roots)
            group_receipts = receipt.get("group_receipts", {})
            def connected_group(group_id):
                group_receipt = group_receipts.get(group_id, {})
                topology = group_receipt.get('topology', {})
                if receipt.get('branch_selection') is None:
                    return bool(topology.get('all_requested_edges_connected', False))
                requested = set(group_receipt.get('selected_edge_ids', ()))
                connected = {edge['edge_id'] for edge in topology.get('edges', ()) if edge.get('connected')}
                return bool(requested) and requested.issubset(connected)
            all_connected = bool(directional_group_ids) and all(connected_group(group_id) for group_id in directional_group_ids)
            components[-1].update(
                run_ids=directional_run_ids, group_ids=directional_group_ids,
                observation_roots=observation_roots,
                connection_status="connected" if all_connected else "not_connected_or_not_assessed",
                connection_assessment_scope=('qualified_selected_branch_union' if receipt.get('branch_selection') is not None
                                             else 'complete_selected_group_union'),
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
    resource_profile: object = None,
    crop_retry_policy: object = None,
    retry_image_provider: object = None,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    """Generate one SAM planning round from immutable detector observations.

    ``runtime`` is a persistent raw-mask tracker. ``image_provider`` is its
    caller-owned immutable native uint8 cache reference; no images survive in
    this function after a bounded independent run. Never pass a bridge-mutated
    volume as ``observation_volume``. The returned merged mmap is caller-owned.
    """
    # Tracking remains owned by the runtime; this hint bounds authenticated
    # independent CPU measurements after all raw proposals have completed.
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
    if resource_profile is not None:
        from .sam_resources import validate_live_sam_resource_profile
        metadata["sam_resource_profile"] = validate_live_sam_resource_profile(resource_profile)
    from .sam_crop_tiling import resolve_sam_crop_mode
    resolved_mode = resolve_sam_crop_mode(crop_mode if crop_mode is not None else
        metadata.get("sam_crop_mode", prepared_plan.crop_mode if prepared_plan is not None else None))
    metadata["sam_crop_mode"] = resolved_mode
    _validate_view(view, wrap_axis, metadata)
    stats["wrap_axis"] = bool(wrap_axis)

    from .sam_evidence import SamEvidenceWriter
    from .sam_policy import select_sam_proposals

    if prepared_plan is None:
        prepared_plan = prepare_sam_interpolation_pass(observation_volume, view=view, scope=metadata,
            pass_index=pass_index, gap_distance=gap_distance, min_radius=min_radius,
            search_angle_deg=search_angle_deg, interpolation_walk_back=interpolation_walk_back,
            interpolation_candidates=interpolation_candidates, interpolation_passes=interpolation_passes,
            wrap_axis=wrap_axis, upstream_lineage=upstream_lineage, spacing_zyx=spacing_zyx,
            planner_limits=planner_limits, canonical_labels=canonical_labels, policy=policy, crop_mode=resolved_mode,
            resource_profile=resource_profile)
    else:
        from .sam_bridge_planning import SAM_CROP_PLANNING_CONTRACT_VERSION
        expected_settings = _planning_identity(scope=metadata, view=view, pass_index=pass_index,
            gap_distance=gap_distance, min_radius=min_radius, search_angle_deg=search_angle_deg,
            interpolation_walk_back=interpolation_walk_back, interpolation_candidates=interpolation_candidates,
            interpolation_passes=interpolation_passes, wrap_axis=wrap_axis, upstream_lineage=upstream_lineage,
            spacing_zyx=spacing_zyx, planner_limits=planner_limits)
        if (prepared_plan.disabled or prepared_plan.settings_sha256 != expected_settings
                or getattr(prepared_plan.plan,"crop_contract_version",None)!=SAM_CROP_PLANNING_CONTRACT_VERSION
                or prepared_plan.observation_buffer_identity != _buffer_identity(observations)
                or prepared_plan.canonical_buffer_identity != (None if canonical_labels is None else
                                                              _buffer_identity(np.asarray(canonical_labels)))):
            raise ValueError("Prepared SAM plan differs from its pinned original observations or planning settings")
        if (observations.flags.writeable and prepared_plan.observation_snapshot_sha256
                and observation_snapshot_sha256(observations) != prepared_plan.observation_snapshot_sha256):
            raise ValueError("Prepared SAM original observation snapshot changed before execution")
    plan = prepared_plan.plan
    if resource_profile is not None and getattr(plan, "contract_lease_budget", None) is not None:
        plan.contract_lease_budget.bind_live_resource_profile(resource_profile)
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
    session_lengths = [len(run.expected_frames) for run in runs]
    stats["sam_session_frame_range"] = ([min(session_lengths), max(session_lengths)] if session_lengths else [0, 0])
    stats["sam_planned_groups"] = sum(group.status == "planned" for group in plan.groups)
    refusal_reasons = {}
    for group in plan.groups:
        if group.status != "planned":
            for reason in group.reasons:
                refusal_reasons[str(reason)] = refusal_reasons.get(str(reason), 0) + 1
    stats["sam_group_refusal_reasons"] = dict(sorted(refusal_reasons.items()))
    from .sam_bridge_planning import SamPlanningLimits
    resolved_limits = planner_limits if planner_limits is not None else SamPlanningLimits()
    stats["sam_group_peak_budget_bytes"] = int(resolved_limits.max_group_bytes)
    stats["sam_contract_resident_budget_bytes"] = int(resolved_limits.max_total_contract_bytes)
    stats["sam_contract_storage_version"] = str(getattr(plan, "contract_storage_version", "eager"))
    stats["sam_original_observation_count"] = len({str(getattr(observation, "original_observation_id", "")
        or observation.observation_id) for observation in plan.observations})
    stats["sam_observation_alias_count"] = sum(getattr(observation, "native_frame_index", None) is not None
        and int(observation.native_frame_index) != int(observation.frame_index) for observation in plan.observations)
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
        image_shape = tuple(getattr(plan, "virtual_shape_tyx", ()) or observations.shape)
        if tuple(int(value) for value in cache_ref.shape) != image_shape:
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
        "planning_contract": str(plan.crop_contract_version),
        "crop_contract_version": str(plan.crop_contract_version),
        "interpolation_min_radius": float(min_radius),
        "planning_status": str(plan.status), "planning_reasons": list(plan.reasons),
        "selection_receipt_required": True,
        "contract_storage_version": str(getattr(plan, "contract_storage_version", "eager")),
        "interpolation_session_contract": "tta_complete_endpoint_interval_without_lta30",
        "evidence_shape_tyx": list(getattr(plan, "virtual_shape_tyx", ()) or observations.shape),
    })
    if getattr(plan, "frame_addressing", None):
        metadata["frame_addressing"] = _plain(plan.frame_addressing)
    if getattr(plan, "contract_lease_budget", None) is not None:
        metadata["contract_resident_budget_bytes"] = int(plan.contract_lease_budget.maximum_bytes)
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
    family_configuration = os.environ.get('YOLO_TTA_SAM_FAMILY_SCHEDULE')
    family_setting = ('fifo' if family_configuration is None else family_configuration).strip().lower()
    if family_setting not in {'flat', 'fifo'}:
        raise SamInterpolationInfrastructureError('YOLO_TTA_SAM_FAMILY_SCHEDULE must be flat or fifo')
    family_capable = callable(getattr(runtime, 'iter_family_results', None))
    if (family_configuration is not None and family_setting == 'fifo' and resolved_mode == 'whole'
            and runs and not family_capable):
        raise SamInterpolationInfrastructureError('Explicit family FIFO dispatch requires a family-aware SAM runtime')
    destination = Path(work_dir) / f"sam_{scope_hash}"
    destination.mkdir(parents=True, exist_ok=True)
    writer = SamEvidenceWriter(destination / "evidence", metadata)
    observation_by_id = {str(observation.observation_id): observation for observation in plan.observations}
    groups = {str(group.group_id): group for group in plan.groups}
    generated_by_index: dict[int, dict[str, object]] = {}
    worker_count = len(getattr(runtime, "device_ids", ())) or 1
    if prepared_plan.cpu_wave_admission and runs:
        worker_count = min(worker_count, int(prepared_plan.cpu_wave_admission['max_in_flight']))
    stats['sam_cpu_wave_admission'] = _plain(prepared_plan.cpu_wave_admission)
    stats['sam_effective_in_flight'] = worker_count if runs else 0
    stats['sam_defer_refill_until_consumed'] = bool(prepared_plan.cpu_wave_admission.get('defer_refill_until_consumed', False))
    execution_order = prepared_plan.execution_order(worker_count)
    tracking_work = prepared_plan.tracker_jobs if resolved_mode == "tiled" else runs
    execution_runs = tuple(tracking_work[index] for index in execution_order)
    family_dispatch = (family_setting == 'fifo' and resolved_mode == 'whole' and bool(runs)
                       and family_capable)
    family_fallback = None
    if family_setting == 'fifo' and not family_dispatch:
        family_fallback = ('tiled_generation_uses_flat_dispatch' if resolved_mode == 'tiled' else
                           'no_tracker_jobs' if not runs else 'runtime_without_family_dispatch')
    family_balance = None
    if family_dispatch and family_configuration is None:
        family_balance = _family_dispatch_balance(runs, execution_order, worker_count)
        if family_balance['fallback_flat']:
            family_dispatch = False
            family_fallback = 'automatic_family_imbalance'
    family_inventory = ()
    family_requested_order = []
    family_submission_receipt = None
    requires_family_submission_receipt = False

    def record_family_submission_order(order):
        nonlocal family_submission_receipt
        if (family_submission_receipt is not None or not isinstance(order, tuple)
                or any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in order)):
            raise SamInterpolationInfrastructureError('SAM family submission order receipt is malformed or repeated')
        family_submission_receipt = tuple(int(index) for index in order)

    if family_dispatch:
        from .sam_tracker_runtime import SamTrackerFamily
        indices_by_family = {}
        for index, run in enumerate(runs):
            indices_by_family.setdefault(str(run.group_id), []).append(index)
        def original_request(index):
            if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
                    or not 0 <= int(index) < len(runs)):
                raise SamInterpolationInfrastructureError('SAM family factory requested an invalid original job index')
            source = _tracker_requests((runs[int(index)],), groups, observation_by_id, cancel_event, resource_profile)
            try:
                request = next(source)
                family_requested_order.append(int(index))
                return request
            finally:
                source.close()
        family_inventory = tuple(SamTrackerFamily(
            family_id=identity, input_indices=tuple(indices),
            run_ids=tuple(str(runs[index].run_id) for index in indices),
            request_factory=original_request,
            frame_work_proxy=sum(len(runs[index].expected_frames) for index in indices))
            for identity, indices in indices_by_family.items())
        stats['sam_effective_in_flight'] = worker_count
    stats["sam_execution_schedule"] = (("bounded_tiled_parent_cohorts" if worker_count>1 else
                                       "bounded_tiled_parent_cohorts_crop_local") if resolved_mode == "tiled" else
                                      "crop_waves" if worker_count > 1 else "group_contiguous")
    stats["sam_execution_order"] = list(execution_order)
    if family_dispatch:
        stats['sam_execution_schedule'] = 'family_fifo'
        stats['sam_execution_order'] = []
        stats['sam_family_completion_order'] = []
    stats['sam_family_schedule_requested'] = family_setting
    stats['sam_family_schedule_explicit'] = family_configuration is not None
    stats['sam_family_schedule_effective'] = 'fifo' if family_dispatch else 'flat'
    stats['sam_family_schedule_fallback_reason'] = family_fallback
    stats['sam_family_dispatch_balance'] = family_balance
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
        if family_dispatch:
            try:
                parameters = inspect.signature(runtime.iter_family_results).parameters
            except (TypeError, ValueError):
                parameters = {}
            declared_receipt = parameters.get('execution_order_callback')
            requires_family_submission_receipt = (declared_receipt is not None
                and declared_receipt.kind != inspect.Parameter.POSITIONAL_ONLY)
            receipt_options = ({'execution_order_callback': record_family_submission_order}
                if requires_family_submission_receipt or any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()) else {})
            result_stream = runtime.iter_family_results(family_inventory,
                source_cache_ref=cache_ref, max_in_flight=worker_count,
                defer_refill_until_consumed=bool(prepared_plan.cpu_wave_admission.get('defer_refill_until_consumed', False)),
                **receipt_options)
        else:
            result_stream = (_iterate_tiled_tracker_results(runtime, prepared_plan, worker_count, groups,
                observation_by_id, cache_ref, cancel_event, resource_profile) if resolved_mode == "tiled" else
                _iterate_tracker_results(runtime, _tracker_requests(execution_runs, groups,
                    observation_by_id, cancel_event, resource_profile), cache_ref, prepared_plan.cpu_wave_admission))
        for execution_index, result in result_stream:
            try:
                if (isinstance(execution_index, bool) or not isinstance(execution_index, (int, np.integer))
                        or not 0 <= int(execution_index) < len(tracking_work)):
                    raise SamInterpolationInfrastructureError("SAM worker returned duplicate or unknown run ownership")
                work_index = int(execution_index) if family_dispatch else execution_order[int(execution_index)]
                if family_dispatch:
                    stats['sam_family_completion_order'].append(work_index)
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
        if family_dispatch:
            # Legacy scripted runtimes may ignore the optional callback. Their
            # passed factory still belongs to this exact scope; another scope
            # cannot overwrite its recorded order after tracker lock handoff.
            scoped_order = (tuple(family_requested_order) if family_submission_receipt is None
                            else family_submission_receipt)
            if (requires_family_submission_receipt and family_submission_receipt is None
                    or len(scoped_order) != len(tracking_work) or set(scoped_order) != set(range(len(tracking_work)))
                    or scoped_order != tuple(family_requested_order)):
                raise SamInterpolationInfrastructureError('SAM family submission order differs from original job ownership')
            stats['sam_execution_order'] = list(scoped_order)
            stats['sam_execution_order_provenance'] = ('scope_owned_request_factory'
                if family_submission_receipt is None else 'completed_iterator_submission_receipt')
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
    if crop_retry_policy is not None and crop_retry_policy.enabled:
        retry_started = time.perf_counter()
        with _trace_sam_phase('retry', metadata.get('scope_id', ''), operation='interpolation'):
            bundle, destination, retry_receipt = _apply_sam_crop_retries(bundle, prepared_plan,
                destination=destination, metadata=metadata, runtime=runtime,
                retry_policy=crop_retry_policy, retry_image_provider=retry_image_provider,
                policy=policy, resource_profile=resource_profile, min_radius=float(min_radius),
                upstream_lineage=upstream_lineage, cancel_event=cancel_event)
        stats['sam_crop_retry'] = retry_receipt
        stats['sam_crop_retry_wall_seconds'] = time.perf_counter()-retry_started
        generated = [dict(run) for run in bundle.runs.values()]
        metadata = _plain(bundle.scope)

    policy_started = time.perf_counter()
    merged_directory = ((Path(runtime_work_dir) / scope_hash) if runtime_work_dir is not None
                        else destination / "_runtime")
    try:
        _check_cancelled(cancel_event)
        with _trace_sam_phase('selection', metadata.get('scope_id', ''), operation='interpolation'):
            receipt = select_sam_proposals(bundle, policy=policy,
                workers=workers,
                **({"resource_profile": resource_profile} if resource_profile is not None else {}))
        stats["sam_policy_wall_seconds"] = time.perf_counter() - policy_started
        (destination / "selection.json").write_text(json.dumps(_plain(receipt), indent=2), encoding="utf-8")
        with _trace_sam_phase('publication', metadata.get('scope_id', ''), operation='interpolation'):
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
        "accepted_connections": (len(receipt['branch_selection']['edges']) if receipt.get('branch_selection') is not None else
                                  sum(len(bundle.groups[group_id].get('edges', ())) for group_id in selected_groups)),
        "default_bridges": sum(int(run["walk_back_index"]) == 0 for run in selected),
        "walk_back_bridges": sum(int(run["walk_back_index"]) > 0 for run in selected),
        "added_voxels": int(added), "bridge_component_count": 2,
        "sam_generated_runs": len(generated), "sam_selected_runs": len(selected),
        "sam_incomplete_runs": sum(not bool(run["complete"]) for run in generated),
        "sam_guarded_rescue": _plain(receipt.get("guarded_rescue", {})),
        "sam_guarded_rescued_groups": len(receipt.get("guarded_rescue", {}).get("rescued_group_ids", ())),
        "sam_guarded_rescued_runs": len(receipt.get("guarded_rescue", {}).get("rescued_run_ids", ())),
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
        "sam_selection_identity": str(receipt.get("selection_identity", "")),
        "sam_selection_resources": _plain(receipt.get("selection_resources", {})),
        "sam_policy_name": str(receipt.get("policy_name", "")),
        "sam_policy_version": int(receipt.get("resolved_policy", {}).get("version", 0)),
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
