"""Production opt-in dynamic LTA windows with sealed predecessor batches.

One completed crop owns its continuation membership. Unrelated in-flight
chains never merge on arrival, so worker completion order cannot change a
crop's model inputs. Escape/split patches run only to the original boundary
and cannot create another patch. All publication still uses LTA's normal
native union, hard-positive restoration and terminal filters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import heapq
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import numpy as np

from .lta_dynamic_crops import (DYNAMIC_CROP_POLICY, DYNAMIC_MODEL_SIDE, DynamicCropSettings,
                                DynamicObject, NativeMask, _bounds_touch_interior_guard,
                                _has_split_components, plan_dynamic_crops,
                                split_dynamic_object, split_depth_budget_reached, touches_interior_guard)
from .lta_propagation import LtaMaskSeed, LtaSeedProvenance, read_seed_artifact, write_seed_artifact
from .lta_scheduler import LtaSessionWork, LtaViewKey
from .lta_tile_tracking import LtaLineageId
from .lta_tiles import TilePlan, _rasterize_source_polygon
from .lta_windows import AnchorDomain, WindowPlan, owned_frame_range, plan_domain_windows
from .lta_workers import LtaWorkerTask


@dataclass(frozen=True)
class DynamicTask:
    work: LtaSessionWork
    payload: dict
    objects: tuple[DynamicObject, ...]
    crop: TilePlan
    window: WindowPlan
    patch: bool
    ordinal: int
    root_id: str
    anchor_domain: AnchorDomain | None = None


class DynamicViewCompletion:
    """Own final view admission after the dynamic queue and workers settle."""

    helper_queue_order = "critical_path"

    def __init__(self, view, device_ids):
        self.view = view
        self.owner = min(device_ids)
        self.settled = False
        self.claim = None

    def owner_for_view(self, view):
        if view != self.view:
            raise ValueError("unknown dynamic LTA view")
        return self.owner

    def claim_backprojection(self, device_id):
        if not self.settled or self.claim is not None or device_id != self.owner:
            return None
        self.claim = SimpleNamespace(owner_device_id=self.owner, view=self.view)
        return self.claim

    def fail_backprojection(self, claim):
        if claim is not self.claim:
            raise ValueError("unknown dynamic backprojection claim")
        self.claim = None

    def complete_backprojection(self, claim):
        if claim is not self.claim:
            raise ValueError("unknown dynamic backprojection claim")


def _task(objects, crop, window, *, patch, ordinal, root_id, view_plan, cache_ref,
          temp_root, conf, empty_frame_limit, settings, plan_order=0, anchor_domain=None):
    from .lta_execution import _tile_payload
    identity = {"policy": DYNAMIC_CROP_POLICY, "root_id": root_id, "patch": patch,
                "ordinal": ordinal, "window": asdict(window), "crop": list(crop.xyxy),
                "seeds": [(item.seed.lineage.token, hashlib.sha256(
                    np.packbits(item.native.mask).tobytes() + repr(item.native.xyxy).encode()).hexdigest())
                          for item in sorted(objects, key=lambda value: value.seed.lineage.token)]}
    token = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    work_id = f"{view_plan.volume_id}::{view_plan.runtime_view_id}::dynamic::{token[:24]}"
    # Geometry-derived IDs remain stable when independent workers finish in a
    # different order. The full geometry is validated on every returned task.
    crop_index = int(hashlib.sha256(repr(crop.xyxy).encode()).hexdigest()[:8], 16)
    seeds = tuple(item.in_crop(crop, object_id=index) for index, item in enumerate(objects))
    artifact = write_seed_artifact(Path(temp_root) / "dynamic-seeds" / f"{token}.npz", seeds)
    start, stop = owned_frame_range(window)
    work = LtaSessionWork(work_id=work_id, view=LtaViewKey(view_plan.volume_id, view_plan.physical_view_id),
                          runtime_view_id=view_plan.runtime_view_id, session_index=plan_order,
                          frame_start=start, frame_stop=stop, plan_order=plan_order,
                          estimated_cost=float(window.frame_count * DYNAMIC_MODEL_SIDE ** 2),
                          projection_key=cache_ref.identity_sha256, tile_index=crop_index,
                          tile_config_id="dynamic", tail_eligible=True, relay_generation=0)
    payload = {"work_id": work_id, "chain_work_id": root_id, "sequence_id":
               f"{view_plan.volume_id}::{view_plan.runtime_view_id}", "cache_ref": cache_ref.payload(),
               "tile_index": crop_index, "tile_config_id": "dynamic", "tile": _tile_payload(crop),
               "neighbors": [], "seed_artifact_path": str(artifact.path),
               "seed_artifact_sha256": artifact.sha256, "window": asdict(window),
               "windows": [asdict(window)], "output_frame_start": start, "output_frame_stop": stop,
               "conf": float(conf), "empty_frame_limit": empty_frame_limit, "relay_generation": 0,
               "relay_min_pixels": 16, "relay_min_probability": 0.5,
               "crop_model_side": DYNAMIC_MODEL_SIDE, "dynamic_crop_policy": DYNAMIC_CROP_POLICY}
    return DynamicTask(work, payload, tuple(objects), crop, window, patch, ordinal, root_id, anchor_domain)


def plan_initial_dynamic_tasks(source, annotations, view_plan, cache_ref, *, temp_root,
                               conf, empty_frame_limit, settings=DynamicCropSettings(), anchor_domains=None):
    tasks, inventory = [], {}
    by_frame = {}
    for annotation in annotations:
        frame = int(annotation.frame_position)
        objects = by_frame.setdefault(frame, [])
        for polygon in annotation.polygons:
            left, top, mask = _rasterize_source_polygon(
                polygon, height=view_plan.frame_height, width=view_plan.frame_width)
            native = NativeMask.from_crop(mask, left=left, top=top)
            if native is None:
                continue
            lineage = LtaLineageId(str(source.volume_id), view_plan.physical_view_id,
                                    view_plan.runtime_view_id,
                                    f"annotation-{frame:06d}-{str(annotation.label_sha256)[:12]}-row-{polygon.row_index:04d}",
                                    tile_config_id="dynamic")
            seed = LtaMaskSeed(lineage, frame, 0, native.mask,
                               source_receipt={"label_sha256": annotation.label_sha256,
                                               "label_row_index": int(polygon.row_index),
                                               "crop_policy": DYNAMIC_CROP_POLICY,
                                               "authoritative_native_bbox": list(native.xyxy)})
            objects.append(DynamicObject(seed, native))
    for frame, objects in sorted(by_frame.items()):
        if not objects:
            continue
        domain = (anchor_domains[frame] if anchor_domains is not None
                  else AnchorDomain(frame, 0, view_plan.frame_count))
        window = plan_domain_windows(domain)[0]
        root_id = f"{view_plan.volume_id}::dynamic-anchor-{frame:06d}"
        for group in plan_dynamic_crops(objects, height=view_plan.frame_height, width=view_plan.frame_width,
                                        settings=settings):
            task = _task(group.objects, group.tile, window, patch=False, ordinal=0, root_id=root_id,
                         view_plan=view_plan, cache_ref=cache_ref, temp_root=temp_root,
                         conf=conf, empty_frame_limit=empty_frame_limit, settings=settings,
                         plan_order=len(tasks), anchor_domain=domain)
            # A crop's fixed predecessor root prevents later arrival batching.
            task = replace(task, root_id=task.work.work_id,
                           payload={**task.payload, "chain_work_id": task.work.work_id})
            tasks.append(task)
            inventory[("dynamic", task.work.tile_index, frame)] = read_seed_artifact(
                task.payload["seed_artifact_path"], expected_sha256=task.payload["seed_artifact_sha256"])
    if not tasks:
        raise ValueError("dynamic LTA requires directly addressable nonempty native annotations")
    return tuple(tasks), inventory


def _read_task_masks(task, manifest, *, settings=DynamicCropSettings()):
    """Validate all evidence, retaining only boundary and earliest event masks.

    The packet remains authoritative for every frame. The coordinator needs
    at most a boundary, a first crop/split event and a split-depth budget event
    per direction and lineage. Validated crop bounds and foreground counts
    identify which other frames need a connected-component check.
    """
    from .lta_coverage import _decode, _read_coverage_packet
    receipt = manifest["lineage_coverage"]
    by_lineage = {item.seed.lineage: item for item in task.objects}
    masks = {}
    scores = manifest.get("crop_observation_scores", {})
    with _read_coverage_packet(
        receipt, expected_work_id=task.work.work_id, tile_index=task.work.tile_index,
        tile_config_id="dynamic", frame_count=task.payload["cache_ref"]["shape"][0],
        frame_start=task.window.frame_start, frame_stop=task.window.frame_stop,
        expected_lineages=by_lineage,
    ) as (_, shape, lineages, rows, _, crop_at):
        if shape != (task.crop.size, task.crop.size):
            raise ValueError("coverage packet shape differs from its dynamic crop")
        by_frame = {}
        for index, row in enumerate(rows):
            by_frame.setdefault(lineages[int(row[0])], {})[int(row[1])] = index

        def tracker_score(lineage, frame):
            score = scores.get(f"{lineage.token}|{frame}")
            if score is None:
                return by_lineage[lineage].seed.tracker_probability
            score = float(score)
            if not 0.0 <= score <= 1.0:
                raise ValueError("tracker_probability must be finite and in [0,1]")
            return score

        def retain_object(index, score, raw=None):
            row = rows[index]
            lineage = lineages[int(row[0])]
            frame = int(row[1])
            if (lineage, frame) in masks:
                return
            if raw is None:
                raw = _decode(crop_at(index))
            native = NativeMask(task.crop.top + int(row[2]), task.crop.left + int(row[3]), raw)
            parent = by_lineage[lineage]
            seed = replace(parent.seed, frame_index=frame, mask=native.mask,
                           provenance=LtaSeedProvenance.TEMPORAL_DOGFOOD,
                           tracker_probability=score,
                           source_receipt={"parent_work_id": task.work.work_id,
                                           "parent_lineage": parent.seed.lineage.token,
                                           "crop_policy": DYNAMIC_CROP_POLICY,
                                           "source_crop_xyxy": list(task.crop.xyxy)})
            masks[(lineage, frame)] = DynamicObject(seed, native, parent.split_depth)

        for lineage, records in by_frame.items():
            for direction in ("forward", "backward"):
                if task.window.direction not in (direction, "both"):
                    continue
                boundary = task.window.frame_stop - 1 if direction == "forward" else task.window.frame_start
                if boundary in records:
                    retain_object(records[boundary], tracker_score(lineage, boundary))
                frames = sorted((frame for frame in records if
                                 (frame > task.window.prompt_frame if direction == "forward"
                                  else frame < task.window.prompt_frame)), reverse=direction == "backward")
                depth_budget_recorded = False
                at_depth_limit = by_lineage[lineage].split_depth >= settings.max_split_depth
                for frame in frames:
                    index = records[frame]
                    y, x, height, width, foreground = map(int, rows[index][2:])
                    left, top = task.crop.left + x, task.crop.top + y
                    edge = _bounds_touch_interior_guard((left, top, left + width, top + height),
                                                        task.crop, guard=settings.guard)
                    score = tracker_score(lineage, frame)
                    check_components = (not depth_budget_recorded if at_depth_limit
                                        else not edge and settings.max_split_children > 1)
                    raw = None
                    split = False
                    if check_components and foreground >= 2 * settings.split_min_pixels:
                        retained = masks.get((lineage, frame))
                        raw = retained.native.mask if retained is not None else _decode(crop_at(index))
                        split = _has_split_components(raw, settings)
                    if split or edge:
                        retain_object(index, score, raw)
                    if edge or (split and not at_depth_limit):
                        break
                    if split:
                        depth_budget_recorded = True
    return masks


def followup_dynamic_tasks(task, masks, *, view_plan, cache_ref, temp_root, conf,
                          empty_frame_limit, settings, audit):
    """Combine edge/split events; patches synchronize at their own boundary."""
    queued = []
    first = task.anchor_domain.frame_start if task.anchor_domain is not None else 0
    limit = task.anchor_domain.frame_stop if task.anchor_domain is not None else view_plan.frame_count
    for direction in ("forward", "backward"):
        if task.window.direction not in (direction, "both"):
            continue
        boundary = task.window.frame_stop - 1 if direction == "forward" else task.window.frame_start
        followup = {}  # (start,stop,prompt,patch) -> fixed-predecessor members
        for parent in task.objects:
            depth_budget_recorded = False
            frames = sorted((frame for lineage, frame in masks if lineage == parent.seed.lineage
                             and (frame > task.window.prompt_frame if direction == "forward"
                                  else frame < task.window.prompt_frame)), reverse=direction == "backward")
            event = None
            for frame in frames:
                item = masks[(parent.seed.lineage, frame)]
                edge = touches_interior_guard(item.native, task.crop, guard=settings.guard)
                pieces = split_dynamic_object(item, settings=settings)
                split = len(pieces) > 1
                if not depth_budget_recorded and split_depth_budget_reached(item, settings=settings):
                    audit["patch_budget_events"].append({"work_id": task.work.work_id,
                        "lineage": item.seed.lineage.token, "frame": frame,
                        "action": "split_depth_limit_combined_foreground_preserved"})
                    depth_budget_recorded = True
                for piece in pieces:
                    if "split_budget" in piece.seed.source_receipt:
                        audit["patch_budget_events"].append({"work_id": task.work.work_id,
                            "lineage": item.seed.lineage.token, "frame": frame,
                            **piece.seed.source_receipt["split_budget"]})
                if edge or split:
                    if task.patch:
                        audit["patch_budget_events"].append({"work_id": task.work.work_id,
                            "lineage": item.seed.lineage.token, "frame": frame,
                            "edge": edge, "split": split, "action": "recorded_no_recursive_patch"})
                        break
                    event = frame, item, pieces, edge, split
                    break
            # A patch must advance time and must improve crop context or split
            # identity. A physical-edge pinned crop has no useful escape retry.
            if event is not None and event[0] != boundary:
                frame, item, pieces, edge, split = event
                crop_groups = plan_dynamic_crops(pieces, height=view_plan.frame_height,
                                                width=view_plan.frame_width, settings=settings)
                moved = any(group.tile.xyxy != task.crop.xyxy for group in crop_groups)
                if split or moved:
                    start, stop = (frame, task.window.frame_stop) if direction == "forward" else (task.window.frame_start, frame + 1)
                    followup.setdefault((start, stop, frame, True), []).extend(pieces)
                    audit["events"].append({"work_id": task.work.work_id, "lineage": item.seed.lineage.token,
                        "frame": frame, "direction": direction, "edge": edge, "split": split,
                        "action": "split_and_expand" if edge and split else "split" if split else "expand",
                        "native_bbox": list(item.native.xyxy)})
                    continue
                audit["patch_budget_events"].append({"work_id": task.work.work_id,
                    "lineage": item.seed.lineage.token, "frame": frame, "edge": edge,
                    "action": "crop_already_at_native_context_limit"})
            item = masks.get((parent.seed.lineage, boundary))
            if item is None:
                audit["empty_boundary_count"] += 1
                continue
            if (direction == "forward" and boundary == limit - 1
                    or direction == "backward" and boundary == first):
                continue
            pieces = split_dynamic_object(item, settings=settings)
            if not depth_budget_recorded and split_depth_budget_reached(item, settings=settings):
                audit["patch_budget_events"].append({"work_id": task.work.work_id,
                    "lineage": item.seed.lineage.token, "frame": boundary,
                    "action": "split_depth_limit_combined_foreground_preserved"})
            if len(pieces) > 1:
                audit["boundary_split_count"] += 1
            start, stop = ((boundary, min(limit, boundary + 30)) if direction == "forward"
                           else (max(first, boundary - 29), boundary + 1))
            followup.setdefault((start, stop, boundary, False), []).extend(pieces)
        for (start, stop, prompt, patch), members in sorted(followup.items()):
            # Patch windows own their prompt frame so repair may add missing
            # geometry there. Normal continuations exclude the shared prompt.
            window = WindowPlan(direction, 0 if patch else task.ordinal + 1, start, stop, prompt,
                                direction, "spatial_relay" if patch else "dogfood")
            for group in plan_dynamic_crops(members, height=view_plan.frame_height,
                                            width=view_plan.frame_width, settings=settings):
                queued.append(_task(group.objects, group.tile, window, patch=patch,
                                    ordinal=task.ordinal if patch else task.ordinal + 1,
                                    root_id=task.root_id, view_plan=view_plan, cache_ref=cache_ref,
                                    temp_root=temp_root, conf=conf, empty_frame_limit=empty_frame_limit,
                                    settings=settings, anchor_domain=task.anchor_domain))
    return tuple(queued)


def drive_dynamic_workers(*, scheduler, pool, initial, view_plan, cache_ref, view_union,
                          temp_root, conf, empty_frame_limit, worker_task_timeout, trace,
                          settings=DynamicCropSettings(), handoff=None):
    from . import lta_execution as execution
    audit = execution._new_worker_audit()
    audit["worker_slots"] = [list(slot) for slot in pool.worker_slots]
    for ready in tuple(getattr(pool, "ready_events", ())):
        execution._accumulate_worker_ready_audit(audit, ready)
    dynamic = {"policy": DYNAMIC_CROP_POLICY, "settings": asdict(settings), "events": [],
               "patch_budget_events": [], "empty_boundary_count": 0, "boundary_split_count": 0,
               "crop_count": 0, "patch_count": 0, "scaled_crop_count": 0,
               "task_receipts": [],
               "membership": "sealed_fixed_predecessor_batches", "recursive_patches": False,
               "motion_extrapolation": False}
    audit["dynamic_crops"] = dynamic
    active, heap, seen, schedule = {}, [], set(), []
    started = {}
    max_tasks = max(1024, sum(len(task.objects) for task in initial) *
                    max(1, (view_plan.frame_count + 28) // 29) * 64)
    dynamic["task_limit"] = max_tasks

    def enqueue(task):
        if task.work.work_id in seen:
            return
        if len(seen) >= max_tasks:
            raise RuntimeError("dynamic LTA exceeded its bounded task budget; result is incomplete")
        seen.add(task.work.work_id)
        first = task.anchor_domain.frame_start if task.anchor_domain is not None else 0
        limit = task.anchor_domain.frame_stop if task.anchor_domain is not None else view_plan.frame_count
        remaining = (limit - task.window.prompt_frame if task.window.direction == "forward"
                     else task.window.prompt_frame - first + 1 if task.window.direction == "backward"
                     else limit - first)
        heapq.heappush(heap, (-remaining, not task.patch, task.work.work_id, task))

    for task in initial:
        enqueue(task)
    while heap or active:
        for slot in pool.worker_slots:
            if slot in active or not heap:
                continue
            _, _, _, task = heapq.heappop(heap)
            attempt = uuid.uuid4().hex
            payload = {**task.payload, "output_dir": str(Path(temp_root) / "dynamic-tasks" /
                        hashlib.sha256(task.work.work_id.encode()).hexdigest()[:24] / attempt),
                       "worker_trace_root": str(trace.path.parent) if trace is not None and trace.path is not None else None}
            worker_task = LtaWorkerTask(task.work.work_id, attempt, "propagation_window", payload)
            pool.submit(worker_task, execution_device_id=slot[0], worker_index=slot[1])
            active[slot] = task, worker_task
            started[slot] = time.monotonic()
            dynamic["crop_count"] += 1
            dynamic["patch_count"] += int(task.patch)
            dynamic["scaled_crop_count"] += int(task.crop.size != DYNAMIC_MODEL_SIDE)
            row = {"work_id": task.work.work_id, "chain_work_id": task.root_id,
                   "plan_order": len(schedule), "execution_device_id": slot[0], "worker_index": slot[1],
                   "frame_start": task.window.frame_start, "frame_stop": task.window.frame_stop,
                   "prompt_frame": task.window.prompt_frame, "direction": task.window.direction,
                   "patch": task.patch, "crop_xyxy": list(task.crop.xyxy), "native_crop_side": task.crop.size,
                   "model_side": DYNAMIC_MODEL_SIDE, "seed_lineages": [item.seed.lineage.token for item in task.objects],
                   "owner_device_id": scheduler.owner, "tail_assist": slot[0] != scheduler.owner,
                   "tile_config_id": "dynamic", "tile_index": task.work.tile_index, "relay_generation": 0}
            if task.anchor_domain is not None:
                row["authoritative_anchor_frame"] = task.anchor_domain.anchor_frame
            schedule.append(row)
            if trace is not None:
                trace.event("dynamic_window_dispatch", **row)
        remaining = min(worker_task_timeout - (time.monotonic() - started[slot]) for slot in active)
        if remaining <= 0:
            raise TimeoutError("dynamic LTA worker exceeded its task lease")
        try:
            result = pool.wait_result(timeout=min(remaining, 10.0))
        except TimeoutError:
            pool.check_liveness()
            continue
        slot = result.execution_device_id, result.worker_index
        if slot not in active:
            raise RuntimeError("dynamic worker completed an unleased slot")
        task, submitted = active.pop(slot)
        started.pop(slot)
        if result.work_id != submitted.work_id or result.attempt_token != submitted.attempt_token:
            raise RuntimeError("dynamic worker result changed its leased identity")
        manifest = execution._load_chain_manifest(result)
        if (manifest["tile"] != task.payload["tile"] or manifest["tile_index"] != task.work.tile_index
                or manifest["window"] != asdict(task.window)
                or tuple(manifest["output_frame_range"]) != owned_frame_range(task.window)):
            raise RuntimeError("dynamic worker changed crop/window coordinates")
        transform = manifest.get("crop_transform", {})
        if (transform.get("native_crop_xyxy") != list(task.crop.xyxy)
                or transform.get("model_shape_hw") != [DYNAMIC_MODEL_SIDE, DYNAMIC_MODEL_SIDE]
                or transform.get("native_mask_shape_hw") != [task.crop.size, task.crop.size]
                or transform.get("prediction_and_dogfood_coordinates") != "native_crop"):
            raise RuntimeError("dynamic worker omitted or changed its native/model crop transform")
        dynamic["task_receipts"].append({"work_id": task.work.work_id,
            "crop_transform": transform, "patch": task.patch,
            "foreground_pixels": int(manifest["foreground_pixels"]),
            "sessions": [{"seed_count": int(window.get("seed_count", 0)),
                          "wall_seconds": window.get("wall_seconds"),
                          "seed_roundtrip_passed": window.get("adapter", {}).get("seed_roundtrip_passed"),
                          "seed_roundtrip_exact": window.get("adapter", {}).get("seed_roundtrip_exact"),
                          "seed_roundtrip_policy": window.get("adapter", {}).get("seed_roundtrip_policy"),
                          "seed_roundtrip_union_metrics": window.get("adapter", {}).get("seed_roundtrip_union_metrics"),
                          "prediction_count": int(window.get("prediction_count", 0)),
                          "dogfood_seed_count": int(window.get("dogfood_seed_count", 0))}
                         for window in manifest.get("windows", ())]})
        masks = _read_task_masks(task, manifest, settings=settings)
        execution._consume_chain_manifest(
            manifest, view_union=view_union, handoff=handoff,
            anchor_frame=(task.anchor_domain.anchor_frame if task.anchor_domain is not None else None),
        )
        execution._accumulate_worker_audit(audit, manifest, result)
        for child in followup_dynamic_tasks(task, masks, view_plan=view_plan, cache_ref=cache_ref,
                                            temp_root=temp_root, conf=conf, empty_frame_limit=empty_frame_limit,
                                            settings=settings, audit=dynamic):
            enqueue(child)
        # Verified worker artifacts are consumed only after their masks have
        # planned every continuation. Remove exact files inside this run root.
        files = [Path(task.payload["seed_artifact_path"]), Path(result.artifact_path),
                 Path(manifest["union"]["path"]), Path(manifest["lineage_coverage"]["path"]),
                 Path(manifest["relay_observation_artifact"]["path"])]
        files.extend(Path(row["path"]) for row in manifest.get("dogfood_seed_artifacts", ()))
        execution._unlink_consumed_temp_artifacts(files, temp_root=Path(temp_root))
        if trace is not None:
            trace.event("dynamic_window_complete", work_id=task.work.work_id,
                        queued=len(heap), active=len(active), events=len(dynamic["events"]))
            trace.flush()
    scheduler.settled = True
    dynamic["task_receipts"].sort(key=lambda row: row["work_id"])
    audit["chain_count"] = len(initial)
    audit["window_graph"] = {"planned_work_count": len(seen), "dispatched_work_count": len(schedule),
                             "dependency_readiness": "verified_predecessor_and_patch_completion",
                             "relay_fan_in": "none_dynamic_native_crops"}
    return 0, dict(pool.pids), tuple(sorted(schedule, key=lambda row: row["work_id"])), audit


__all__ = ("DynamicTask", "DynamicViewCompletion", "plan_initial_dynamic_tasks",
           "followup_dynamic_tasks", "drive_dynamic_workers")
