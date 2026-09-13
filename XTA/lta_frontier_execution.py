"""Canonical temporal frontier execution using bounded, deterministic worker waves."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time


CANONICAL_FRONTIER_POLICY = "canonical_temporal_frontier/1"
FRONTIER_WAVE_SEEDS = 32


def _validate_frontier_observation_bindings(observations, payload, input_seeds):
    """Bind decoded relay evidence to its originating task and seed identities."""
    seeds_by_lineage = {seed.lineage: seed for seed in input_seeds}
    neighbors = {int(record["tile_index"]): record["tile"] for record in payload["neighbors"]}
    window = payload["window"]
    start, stop = int(window["frame_start"]), int(window["frame_stop"])
    for record in observations.values():
        source = seeds_by_lineage.get(record["lineage"])
        destination = neighbors.get(int(record["destination_index"]))
        if source is None or destination is None:
            raise RuntimeError("frontier relay observation changed its input lineage or neighbor destination")
        observed_source = record["seed"]
        if (observed_source.object_id != source.object_id
                or tuple(observed_source.visited_tile_indices) != tuple(source.visited_tile_indices)):
            raise RuntimeError("frontier relay observation changed its source object identity")
        shape = (int(destination["size"]), int(destination["size"]))
        for endpoints in record["episodes"]:
            for frame, _packed, observed_shape, _probability in endpoints:
                if not start <= int(frame) < stop or tuple(observed_shape) != shape:
                    raise RuntimeError("frontier relay observation changed its frame range or neighbor geometry")


def _drive_canonical_relay_frontier(
    records, *, pool, view_plan, cache_ref, view_union, coverage, temp_root,
    conf, empty_frame_limit, worker_task_timeout, trace, worker_audit,
    dispatched, first_plan_order, max_relay_generations,
    device_ids=None, workers_per_device=None,
):
    """Advance merged relay masks one canonical slab at a time.

    Temporal mailboxes persist across spatial generations. A work identity has
    no originating route or chain, and another attempt requires new input
    foreground. All results in a fixed logical wave are validated before any
    continuation is offered; device count and completion order cannot change
    which input revisions belong to a wave.
    """
    from . import lta_execution as execution
    from .lta_frontier import CanonicalFrontier
    from .lta_postprocessing import fill_binary_mask_holes_2d
    from .lta_propagation import partition_mask_seed_sessions, read_seed_artifact, write_seed_artifact
    from .lta_relay_episodes import read_relay_observations, merge_relay_observations_across_chains
    from .lta_scheduler import LtaSessionWork, LtaViewAffinityScheduler, LtaViewKey
    from .lta_worker_adapter import _write_relay_artifacts

    temp_root = Path(temp_root)
    devices = tuple(device_ids if device_ids is not None else getattr(pool, "device_ids", tuple(pool.pids)))
    slots = int(workers_per_device if workers_per_device is not None else getattr(pool, "workers_per_device", 1))
    grids = {grid.config_id: grid for grid in view_plan.tile_grids}
    view_key = LtaViewKey(view_plan.volume_id, view_plan.physical_view_id)
    audit = {
        "policy": CANONICAL_FRONTIER_POLICY, "wave_seed_limit": FRONTIER_WAVE_SEEDS,
        "waves": 0, "tasks": 0, "maximum_wave_tasks": 0,
        "spatial_seed_candidates": 0, "spatial_seed_handoffs": 0,
        "boundary_seed_handoffs": 0, "generations": [],
    }
    worker_audit["canonical_frontier"] = audit
    next_order = int(first_plan_order)
    incoming = tuple(records)
    last_generation = 0

    def event(name, **fields):
        if trace is not None:
            trace.event(name, **fields)
            trace.flush()

    def phase(name, **fields):
        return nullcontext() if trace is None else trace.phase(name, **fields)

    def contained(path):
        resolved = Path(path).resolve(strict=True)
        try:
            execution._artifact_ownership_path(resolved).relative_to(
                execution._artifact_ownership_path(temp_root.resolve(strict=True)))
        except ValueError as error:
            raise RuntimeError("frontier artifact is outside its scratch root") from error
        return resolved

    with CanonicalFrontier(temp_root / "canonical_frontier.sqlite3", frame_count=view_plan.frame_count) as frontier:
        for generation in range(1, int(max_relay_generations) + 2):
            generation_tasks = 0
            generation_waves = 0
            generation_observations = {}
            with phase("frontier_spatial_admission", generation=generation, relay_record_count=len(incoming)):
                # Authenticate every offered artifact before changing mailbox state.
                paths = execution._verified_relay_artifact_paths(incoming)
                grouped = defaultdict(list)
                for record, path in zip(incoming, paths):
                    contained(path)
                    lineage = execution._lineage_from_record(record["lineage"])
                    tile_index = int(record["destination_tile_index"])
                    direction = str(record["temporal_direction"])
                    frame = int(record["frame_index"])
                    grid = grids.get(lineage.tile_config_id)
                    seeds = read_seed_artifact(path, expected_sha256=str(record["seed_artifact_sha256"]))
                    if (grid is None or not 0 <= tile_index < len(grid.tiles)
                            or direction not in {"forward", "backward"}
                            or not 0 <= frame < view_plan.frame_count
                            or lineage.volume_id != view_plan.volume_id
                            or lineage.physical_view_id != view_plan.physical_view_id
                            or lineage.runtime_view_id != view_plan.runtime_view_id
                            or len(seeds) != 1 or seeds[0].lineage != lineage
                            or seeds[0].frame_index != frame
                            or tuple(seeds[0].mask.shape) != (grid.tile_size, grid.tile_size)):
                        raise RuntimeError("relay seed metadata does not match its frontier destination")
                    grouped[(lineage, tile_index, frame, direction)].append(record)
                for (_lineage, tile_index, _frame, direction), group in sorted(grouped.items()):
                    seed = execution._select_and_merge_relay_group(
                        group, direction=direction, destination_tile_index=tile_index, generation=generation)
                    audit["spatial_seed_candidates"] += 1
                    # Join historical input support before handoff. A fragment
                    # covered on its own can close a hole in the accumulated seed.
                    frontier.offer(seed, tile_index=tile_index, direction=direction)
                execution._unlink_consumed_temp_artifacts(paths, temp_root=temp_root)
            incoming = ()
            entries = frontier.take_wave(limit=FRONTIER_WAVE_SEEDS)
            if not entries:
                audit["frontier"] = frontier.stats()
                event("frontier_complete", generation=last_generation, **audit["frontier"])
                print(f"LTA canonical frontier settled: spatial_generation={last_generation} tasks={audit['tasks']}", flush=True)
                return last_generation
            if generation > int(max_relay_generations):
                raise RuntimeError("LTA canonical frontier has new support beyond its spatial-generation safety bound")

            while entries:
                audit["waves"] += 1
                generation_waves += 1
                wave_index = int(audit["waves"])
                seed_groups = defaultdict(list)
                for entry in entries:
                    seed = replace(entry.seed, mask=fill_binary_mask_holes_2d(entry.seed.mask), relay_generation=generation)
                    if coverage.can_handoff(seed, tile_index=entry.tile_index, direction=entry.direction):
                        audit["boundary_seed_handoffs"] += 1
                        if "spatial_relay" in seed.source_receipt.get("input_provenances", ()):
                            audit["spatial_seed_handoffs"] += 1
                        continue
                    seed = coverage.merge_prompt_support(seed, tile_index=entry.tile_index)
                    seed = replace(seed, mask=fill_binary_mask_holes_2d(seed.mask))
                    window = entry.window
                    group_key = (seed.lineage.tile_config_id, entry.tile_index, entry.direction,
                                 window.frame_start, window.frame_stop, window.prompt_frame)
                    seed_groups[group_key].append((entry, seed))
                tasks = []
                for (config_id, tile_index, direction, _start, _stop, _prompt), values in sorted(seed_groups.items()):
                    grid = grids[config_id]
                    window = values[0][0].window
                    seed_keys = {id(seed): entry for entry, seed in values}
                    ordered_seeds = tuple(seed for _entry, seed in sorted(values, key=lambda pair: pair[1].lineage))
                    for batch in partition_mask_seed_sessions(ordered_seeds):
                        revision = hashlib.sha256("|".join(
                            f"{seed_keys[id(seed)].key}:{seed_keys[id(seed)].revision_sha256}:"
                            f"{hashlib.sha256(seed.mask.tobytes()).hexdigest()}" for seed in batch
                        ).encode()).hexdigest()
                        work_id = f"{view_plan.volume_id}::{view_plan.runtime_view_id}::{config_id}::tile-{tile_index:04d}::frontier-{revision}"
                        normalized = tuple(replace(seed, object_id=index) for index, seed in enumerate(batch))
                        artifact = write_seed_artifact(
                            temp_root / "frontier-seeds" / f"{revision}.npz", normalized)
                        payload = execution._base_chain_payload(
                            work_id=work_id, view_plan=view_plan, grid=grid, tile_index=tile_index,
                            cache_ref=cache_ref, seed_path=artifact.path, seed_sha256=artifact.sha256,
                            windows=(window,), output_frame_start=window.frame_start,
                            output_frame_stop=window.frame_stop, conf=conf,
                            empty_frame_limit=empty_frame_limit, relay_generation=generation)
                        payload.update(chain_work_id=work_id, chain_window_count=1, window_index=0,
                                       window=asdict(window), predecessor_work_id=None,
                                       frontier_wave=wave_index, frontier_revision=revision)
                        if getattr(trace, "path", None) is not None:
                            payload["worker_trace_root"] = str(Path(trace.path).parent)
                        # A scheduler owns only this sealed wave. Spatial-generation
                        # metadata belongs to the controller and task payload.
                        work = LtaSessionWork(
                            work_id=work_id, view=view_key, runtime_view_id=view_plan.runtime_view_id,
                            session_index=next_order, frame_start=window.frame_start,
                            frame_stop=window.frame_stop, plan_order=next_order,
                            estimated_cost=float(window.frame_count * grid.tile_size ** 2),
                            projection_key=cache_ref.identity_sha256, tile_index=tile_index,
                            tile_config_id=config_id, tail_eligible=True, relay_generation=0)
                        tasks.append(execution._PlannedChain(work, payload))
                        next_order += 1
                audit["maximum_wave_tasks"] = max(audit["maximum_wave_tasks"], len(tasks))
                event("frontier_wave_start", generation=generation, wave=wave_index,
                      input_revisions=len(entries), planned_windows=len(tasks))
                received = {}
                if tasks:
                    scheduler = LtaViewAffinityScheduler(
                        (task.work for task in tasks), devices, workers_per_device=slots,
                        helper_queue_order="head", max_relay_generation=0)
                    scheduler.mark_projection_ready(view_key, device_id=scheduler.owner_for_view(view_key))
                    payloads = {task.work.work_id: task.payload for task in tasks}
                    active, started = {}, {}
                    while len(received) < len(tasks):
                        execution._dispatch_available(
                            scheduler, pool, payloads, active, started, dispatched,
                            tasks_root=temp_root / "frontier-tasks", trace=trace,
                            generation_override=generation)
                        if not active:
                            raise RuntimeError("canonical frontier wave has no active or ready work")
                        oldest = min(started, key=started.get)
                        remaining = float(worker_task_timeout) - (time.monotonic() - started[oldest])
                        if remaining <= 0:
                            raise TimeoutError(f"frontier worker task {oldest!r} exceeded its lease")
                        with phase("wait_for_window_result", generation=generation, wave=wave_index,
                                   **scheduler.queue_counts()):
                            try:
                                result = pool.wait_result(timeout=min(remaining, 10.0))
                            except TimeoutError:
                                pool.check_liveness()
                                continue
                        claim = active.pop(result.work_id, None)
                        started.pop(result.work_id, None)
                        if claim is None or (int(result.execution_device_id), int(getattr(result, "worker_index", 0))) != (
                                claim.execution_device_id, claim.worker_index):
                            raise RuntimeError("frontier result does not match its leased worker slot")
                        payload = payloads[result.work_id]
                        with phase("window_result_reduction", work_id=result.work_id, generation=generation):
                            manifest = execution._load_chain_manifest(result)
                            if (manifest["output_frame_range"] != [payload["output_frame_start"], payload["output_frame_stop"]]
                                    or manifest.get("tile_index") != payload["tile_index"]
                                    or manifest.get("tile") != payload["tile"]
                                    or ("tile_config_id" in manifest
                                        and manifest["tile_config_id"] != payload["tile_config_id"])
                                    or manifest.get("relay_generation") != generation):
                                raise RuntimeError("frontier worker changed its task coordinates")
                            observation = manifest.get("relay_observation_artifact")
                            if observation is None or manifest.get("lineage_coverage") is None:
                                raise RuntimeError("canonical frontier requires per-lineage worker evidence")
                            contained(observation["path"])
                            observations = read_relay_observations(observation)
                            input_seeds = read_seed_artifact(
                                payload["seed_artifact_path"], expected_sha256=payload["seed_artifact_sha256"])
                            _validate_frontier_observation_bindings(observations, payload, input_seeds)
                            outgoing = execution._verified_dogfood_artifacts(manifest, payload)
                            window = payload["window"]
                            endpoints = {int(window["frame_start"]), int(window["frame_stop"]) - 1}
                            if any(frame not in endpoints for frame in outgoing):
                                raise RuntimeError("frontier dogfood lies outside a window boundary")
                            for record in outgoing.values():
                                contained(record["path"])
                            legacy_paths = execution._verified_relay_artifact_paths(tuple(manifest.get("relays", ())))
                            for path in legacy_paths:
                                contained(path)
                            _unused_relays, union_path = execution._consume_chain_manifest(manifest, view_union=view_union)
                        with phase("lineage_coverage_ingest", work_id=result.work_id, generation=generation):
                            execution._ingest_lineage_coverage(manifest, payload, coverage, temp_root=temp_root)
                        execution._unlink_consumed_temp_artifacts((union_path,), temp_root=temp_root)
                        received[result.work_id] = (result, manifest, observations, outgoing, legacy_paths)
                        scheduler.complete(claim, result)
                    scheduler.drain_committable()
                    scheduler.seal_view(view_key)

                # Barrier: no result from this wave may create another attempt
                # before every result and coverage receipt has passed validation.
                frontier.complete(entries)
                for task in tasks:
                    result, manifest, observations, outgoing, legacy_paths = received[task.work.work_id]
                    payload = task.payload
                    execution._accumulate_worker_audit(worker_audit, manifest, result)
                    worker_audit["chain_count"] = int(worker_audit["chain_count"]) + 1
                    target = generation_observations.setdefault((payload["tile_config_id"], payload["tile_index"]), {})
                    merge_relay_observations_across_chains(target, observations)
                    window = payload["window"]
                    direction = window["direction"]
                    boundary = int(window["frame_stop"]) - 1 if direction == "forward" else int(window["frame_start"])
                    terminal = boundary == (view_plan.frame_count - 1 if direction == "forward" else 0)
                    if not terminal and boundary in outgoing:
                        record = outgoing[boundary]
                        for seed in read_seed_artifact(record["path"], expected_sha256=record["sha256"]):
                            frontier.offer(seed, tile_index=payload["tile_index"], direction=direction)
                    consumed = [Path(result.artifact_path), Path(payload["seed_artifact_path"]),
                                Path(manifest["relay_observation_artifact"]["path"]),
                                *legacy_paths, *(Path(record["path"]) for record in outgoing.values())]
                    execution._unlink_consumed_temp_artifacts(consumed, temp_root=temp_root)
                generation_tasks += len(tasks)
                audit["tasks"] += len(tasks)
                event("frontier_wave_complete", generation=generation, wave=wave_index,
                      executed_windows=len(tasks))
                entries = frontier.take_wave(limit=FRONTIER_WAVE_SEEDS)

            last_generation = generation
            emitted = []
            with phase("generation_relay_emission", generation=generation):
                for (config_id, tile_index), observations in sorted(generation_observations.items()):
                    emitted.extend(_write_relay_artifacts(
                        observations, output_dir=temp_root / "frontier-relays" / f"generation-{generation:04d}" /
                        execution._safe_token(config_id) / f"tile-{tile_index:04d}",
                        source_tile_index=tile_index, generation=generation))
            worker_audit["relay_artifact_count"] = int(worker_audit["relay_artifact_count"]) + len(emitted)
            summary = {"generation": generation, "waves": generation_waves,
                       "executed_windows": generation_tasks, "emitted_endpoints": len(emitted),
                       "frontier": frontier.stats(), "coverage": coverage.stats()}
            audit["generations"].append(summary)
            event("canonical_generation_summary", **summary)
            print(f"LTA canonical frontier: spatial_generation={generation} windows={generation_tasks} "
                  f"waves={generation_waves} outgoing_endpoints={len(emitted)}", flush=True)
            incoming = tuple(emitted)
    raise AssertionError("frontier safety-bound loop did not terminate")
