"""Generation-based relay reference backend for tests and developer comparisons.

Production always uses the canonical temporal frontier. This backend preserves
older generation admission and mask revision semantics while sharing the same
bounded-window execution engine and artifact validation as production.
"""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from XTA import lta_execution as coordinator
from XTA.lta_execution import _PlannedChain
from XTA.lta_propagation import LtaMaskSeed, partition_mask_seed_sessions, write_seed_artifact
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.lta_runtime import LtaRuntimeViewPlan
from XTA.lta_scheduler import LtaSessionWork, LtaSpatialRelayKey, LtaViewAffinityScheduler, LtaViewKey
from XTA.lta_windows import plan_directional_windows
from XTA.lta_workers import LtaWorkerPool
from XTA.lta_window_execution import RelayContinuation, WindowExecution, drive_window_generations


@dataclass(frozen=True)
class _RelayMaskRevision:
    shape_hw: tuple[int, int]
    packed_mask: Any
    revision_sha256: str
    visited_tile_indices: tuple[int, ...]
    tracker_probability: float


def _relay_mask_revision(mask: object) -> tuple[tuple[int, int], object, str]:
    import numpy as np

    binary = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    if binary.ndim != 2 or not bool(binary.any()):
        raise ValueError("a relay mask revision must contain 2D foreground")
    shape = tuple(int(value) for value in binary.shape)
    packed = np.ascontiguousarray(np.packbits(binary.reshape(-1)), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(f"{shape[0]}x{shape[1]}:".encode("ascii"))
    digest.update(memoryview(packed).cast("B"))
    packed.setflags(write=False)
    return shape, packed, digest.hexdigest()


def _accumulate_relay_seed_revision(
    event_key: LtaSpatialRelayKey,
    seed: LtaMaskSeed,
    revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
    *, enable_2D_hole_fill: bool = False,
) -> tuple[LtaMaskSeed, str] | None:
    """Admit only new foreground and return the accumulated event seed."""

    import numpy as np

    if event_key.mask_revision_sha256 != "0" * 64:
        raise ValueError("relay event keys must use the zero revision placeholder")
    shape, current_packed, _current_digest = _relay_mask_revision(seed.mask)
    previous = revisions.get(event_key)
    previous_digest = None
    if previous is None:
        accumulated_packed = np.asarray(current_packed, dtype=np.uint8).copy()
        delta_pixels = int(np.unpackbits(accumulated_packed).sum())
        visited = seed.visited_tile_indices
        probability = seed.tracker_probability
    else:
        if previous.shape_hw != shape or previous.packed_mask.shape != current_packed.shape:
            raise RuntimeError("one relay event changed mask shape across generations")
        current_array = np.asarray(current_packed, dtype=np.uint8)
        previous_array = np.asarray(previous.packed_mask, dtype=np.uint8)
        delta = np.bitwise_and(current_array, np.bitwise_not(previous_array))
        delta_pixels = int(np.unpackbits(delta).sum())
        if delta_pixels == 0:
            return None
        accumulated_packed = np.bitwise_or(previous_array, current_array)
        previous_digest = previous.revision_sha256
        visited = tuple(
            sorted(set(previous.visited_tile_indices) | set(seed.visited_tile_indices))
        )
        probability = max(previous.tracker_probability, seed.tracker_probability)

    accumulated_mask = np.unpackbits(
        accumulated_packed,
        count=shape[0] * shape[1],
    ).reshape(shape)
    filled = coordinator.fill_merged_seed_mask_holes_2d(
        accumulated_mask,
        enabled=enable_2D_hole_fill,
        context={
            "destination_tile_index": event_key.destination_tile_index,
            "temporal_direction": event_key.temporal_direction,
            "frame_index": event_key.frame_index,
            "relay_generation": seed.relay_generation,
            "accumulated_revision": True,
        },
    )
    shape, packed, digest = _relay_mask_revision(filled.mask)
    revisions[event_key] = _RelayMaskRevision(
        shape_hw=shape,
        packed_mask=packed,
        revision_sha256=digest,
        visited_tile_indices=tuple(visited),
        tracker_probability=float(probability),
    )
    accumulated_pixels = int(np.unpackbits(np.asarray(packed, dtype=np.uint8)).sum())
    return (
        LtaMaskSeed(
            lineage=seed.lineage,
            frame_index=seed.frame_index,
            object_id=seed.object_id,
            mask=filled.mask,
            provenance=seed.provenance,
            tracker_probability=float(probability),
            relay_generation=seed.relay_generation,
            visited_tile_indices=tuple(visited),
            source_receipt={
                **dict(seed.source_receipt),
                "relay_mask_revision_sha256": digest,
                "previous_relay_mask_revision_sha256": previous_digest,
                "new_foreground_pixels": delta_pixels,
                "accumulated_foreground_pixels": accumulated_pixels,
                "accumulated_hole_fill": filled.receipt.manifest_record(),
            },
        ),
        digest,
    )


def _prime_authoritative_relay_destinations(
    scheduler: LtaViewAffinityScheduler,
    view: LtaViewKey,
    seed_inventory: Mapping[tuple[str, int, int], Sequence[LtaMaskSeed]],
    relay_mask_revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
) -> None:
    """Prime event masks so identical relays cannot reopen authoritative seeds."""

    for (config_id, tile_index, _frame), seeds in seed_inventory.items():
        for seed in seeds:
            for direction in ("forward", "backward"):
                event_key = LtaSpatialRelayKey(
                    view=view,
                    runtime_view_id=seed.lineage.runtime_view_id,
                    tile_config_id=config_id,
                    lineage_id=seed.lineage.lineage_id,
                    destination_tile_index=tile_index,
                    frame_index=seed.frame_index,
                    temporal_direction=direction,
                )
                shape, packed, revision = _relay_mask_revision(seed.mask)
                relay_mask_revisions[event_key] = _RelayMaskRevision(
                    shape_hw=shape,
                    packed_mask=packed,
                    revision_sha256=revision,
                    visited_tile_indices=seed.visited_tile_indices,
                    tracker_probability=seed.tracker_probability,
                )
                scheduler.admit_spatial_relay(
                    replace(event_key, mask_revision_sha256=revision)
                )


def _plan_relay_generation(
    records: Sequence[Mapping[str, object]],
    *,
    generation: int,
    view_plan: LtaRuntimeViewPlan,
    cache_ref: LtaPhysicalViewCacheRef,
    scheduler: LtaViewAffinityScheduler,
    relay_mask_revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
    temp_root: Path,
    conf: float,
    empty_frame_limit: int | None,
    first_plan_order: int,
    coverage_ledger: object | None = None,
    admission_audit: dict[str, object] | None = None,
    enable_2D_hole_fill: bool = False,
) -> tuple[_PlannedChain, ...]:
    # Validate every emitted relay, including arrivals that an idempotency key
    # will deliberately discard.  Cleanup below is therefore restricted to
    # exact artifacts whose worker receipts have been proven.
    relay_artifacts = coordinator._verified_relay_artifact_paths(records)
    grid_by_id = {grid.config_id: grid for grid in view_plan.tile_grids}
    by_relay_key: dict[LtaSpatialRelayKey, list[Mapping[str, object]]] = {}
    view_key = LtaViewKey(view_plan.volume_id, view_plan.physical_view_id)
    for record in records:
        lineage = coordinator._lineage_from_record(record["lineage"])  # type: ignore[arg-type]
        destination_tile_index = int(record["destination_tile_index"])
        frame_index = int(record["frame_index"])
        key = LtaSpatialRelayKey(
            view=view_key,
            runtime_view_id=lineage.runtime_view_id,
            tile_config_id=lineage.tile_config_id,
            lineage_id=lineage.lineage_id,
            destination_tile_index=destination_tile_index,
            frame_index=frame_index,
            temporal_direction=str(record["temporal_direction"]),
        )
        by_relay_key.setdefault(key, []).append(record)

    accepted: list[tuple[LtaSpatialRelayKey, LtaMaskSeed]] = []
    for event_key in sorted(by_relay_key):
        seed = coordinator._select_and_merge_relay_group(
            by_relay_key[event_key],
            direction=event_key.temporal_direction,
            destination_tile_index=event_key.destination_tile_index,
            generation=generation,
            enable_2D_hole_fill=enable_2D_hole_fill,
        )
        if admission_audit is not None:
            admission_audit["spatial_seed_candidates"] = int(admission_audit.get("spatial_seed_candidates", 0)) + 1
        if coverage_ledger is not None and coverage_ledger.can_handoff(
            seed, tile_index=event_key.destination_tile_index,
            direction=event_key.temporal_direction,
        ):
            if admission_audit is not None:
                admission_audit["spatial_seed_handoffs"] = int(admission_audit.get("spatial_seed_handoffs", 0)) + 1
            continue
        accumulated = _accumulate_relay_seed_revision(
            event_key,
            seed,
            relay_mask_revisions,
            enable_2D_hole_fill=enable_2D_hole_fill,
        )
        if accumulated is None:
            if admission_audit is not None:
                admission_audit["repeated_seed_revisions"] = int(admission_audit.get("repeated_seed_revisions", 0)) + 1
            continue
        accumulated_seed, revision = accumulated
        key = replace(event_key, mask_revision_sha256=revision)
        if not scheduler.admit_spatial_relay(key):
            raise RuntimeError("a new monotone relay revision reused an admitted digest")
        accepted.append((key, accumulated_seed))

    # Multiplex lineages that share a destination, prompt, and direction.
    grouped: dict[tuple[str, int, int, str], list[tuple[LtaSpatialRelayKey, LtaMaskSeed]]] = {}
    for key, seed in accepted:
        group_key = (
            key.tile_config_id,
            key.destination_tile_index,
            seed.frame_index,
            key.temporal_direction,
        )
        grouped.setdefault(group_key, []).append((key, seed))

    planned: list[_PlannedChain] = []
    plan_order = int(first_plan_order)
    for (config_id, tile_index, prompt_frame, direction), values in sorted(grouped.items()):
        grid = grid_by_id.get(config_id)
        if grid is None:
            raise RuntimeError(f"relay references unknown tile configuration {config_id}")
        windows = plan_directional_windows(
            frame_start=0,
            frame_stop=view_plan.frame_count,
            prompt_frame=prompt_frame,
            direction=direction,
        )
        work_id_prefix = (
            f"{view_plan.volume_id}::{view_plan.runtime_view_id}::{config_id}::"
            f"tile-{tile_index:04d}::relay-{generation:04d}::{direction}::"
            f"frame-{prompt_frame:06d}"
        )
        output_start = prompt_frame if direction == "forward" else 0
        output_stop = view_plan.frame_count if direction == "forward" else prompt_frame + 1
        tile = grid.tiles[tile_index]
        ordered_values = tuple(sorted(values, key=lambda item: item[0].lineage_id))
        relay_key_by_seed_identity = {
            id(seed): key for key, seed in ordered_values
        }
        if len(relay_key_by_seed_identity) != len(ordered_values):
            raise RuntimeError("one relay seed object was associated with multiple relay keys")
        value_batches = tuple(
            tuple(
                (relay_key_by_seed_identity[id(seed)], seed)
                for seed in seed_batch
            )
            for seed_batch in partition_mask_seed_sessions(
                tuple(seed for _key, seed in ordered_values)
            )
        )
        batch_count = len(value_batches)
        for batch_index, batch in enumerate(value_batches):
            normalized_seeds = tuple(
                LtaMaskSeed(
                    lineage=seed.lineage,
                    frame_index=seed.frame_index,
                    object_id=object_id,
                    mask=seed.mask,
                    provenance=seed.provenance,
                    tracker_probability=seed.tracker_probability,
                    relay_generation=seed.relay_generation,
                    visited_tile_indices=seed.visited_tile_indices,
                    source_receipt=seed.source_receipt,
                )
                for object_id, (_key, seed) in enumerate(batch)
            )
            work_id = (
                f"{work_id_prefix}::"
                f"batch-{batch_index:04d}-of-{batch_count:04d}"
            )
            seed_artifact = write_seed_artifact(
                temp_root
                / "seeds"
                / f"generation-{generation:04d}"
                / f"{hashlib.sha256(work_id.encode()).hexdigest()[:20]}.npz",
                normalized_seeds,
            )
            work = LtaSessionWork(
                work_id=work_id,
                view=view_key,
                runtime_view_id=view_plan.runtime_view_id,
                session_index=plan_order,
                frame_start=output_start,
                frame_stop=output_stop,
                plan_order=plan_order,
                estimated_cost=float((output_stop - output_start) * tile.size * tile.size),
                projection_key=cache_ref.identity_sha256,
                tile_index=tile_index,
                tile_config_id=config_id,
                tail_eligible=True,
                relay_generation=generation,
            )
            payload = coordinator._base_chain_payload(
                work_id=work_id,
                view_plan=view_plan,
                grid=grid,
                tile_index=tile_index,
                cache_ref=cache_ref,
                seed_path=seed_artifact.path,
                seed_sha256=seed_artifact.sha256,
                windows=windows,
                output_frame_start=output_start,
                output_frame_stop=output_stop,
                conf=conf,
                empty_frame_limit=empty_frame_limit,
                relay_generation=generation,
            )
            planned.append(_PlannedChain(work=work, payload=payload))
            plan_order += 1
    coordinator._unlink_consumed_temp_artifacts(relay_artifacts, temp_root=temp_root)
    return tuple(planned)


def _continue_relay_generations(execution, settled, *, relay_mask_revisions):
    """Plan the next old-style relay generation after shared graph settlement."""
    scheduler = execution.scheduler
    trace = execution.trace
    generation = settled.generation
    coverage = settled.coverage

    def plan_relays(next_generation):
        records = settled.records
        phase = trace.phase("relay_planning", generation=next_generation, relay_record_count=len(records)) if trace is not None else nullcontext()
        with phase:
            return _plan_relay_generation(
                records, generation=next_generation,
                view_plan=execution.view_plan, cache_ref=execution.cache_ref,
                scheduler=scheduler, relay_mask_revisions=relay_mask_revisions,
                temp_root=execution.temp_root, conf=execution.conf,
                empty_frame_limit=execution.empty_frame_limit,
                first_plan_order=settled.first_plan_order,
                coverage_ledger=coverage, admission_audit=settled.admission_audit,
                enable_2D_hole_fill=execution.enable_2D_hole_fill,
            )

    if generation >= scheduler.max_relay_generation:
        overflow = plan_relays(generation + 1)
        if overflow:
            raise RuntimeError(
                "LTA spatial relay did not reach a mask fixed point within "
                f"the safety bound of {scheduler.max_relay_generation} generations"
            )
        return RelayContinuation(generation)
    next_generation = generation + 1
    planned = plan_relays(next_generation)
    if not planned:
        if trace is not None:
            trace.event("relay_generation_planned", generation=next_generation,
                        accepted_chains=0, planned_windows=0, coverage=coverage.stats())
            trace.flush()
        print(f"LTA relay fixed point: generation={generation} accepted_next_chains=0", flush=True)
        return RelayContinuation(generation)
    return RelayContinuation(next_generation, planned)


def _drive_workers_to_fixed_point(
    *,
    scheduler: LtaViewAffinityScheduler,
    pool: LtaWorkerPool,
    initial: Sequence[_PlannedChain],
    view_plan: LtaRuntimeViewPlan,
    cache_ref: LtaPhysicalViewCacheRef,
    view_union: object,
    relay_mask_revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
    temp_root: Path,
    conf: float,
    empty_frame_limit: int | None,
    worker_task_timeout: float,
    trace: object | None = None,
    enable_2D_hole_fill: bool = False,
):
    """Execute the generation-based reference using the shared window engine."""
    return drive_window_generations(
        execution=WindowExecution(
            scheduler=scheduler, pool=pool, view_plan=view_plan, cache_ref=cache_ref,
            view_union=view_union, temp_root=temp_root, conf=conf,
            empty_frame_limit=empty_frame_limit, worker_task_timeout=worker_task_timeout,
            trace=trace, enable_2D_hole_fill=enable_2D_hole_fill,
        ),
        initial=initial,
        continue_relays=lambda execution, settled: _continue_relay_generations(
            execution, settled, relay_mask_revisions=relay_mask_revisions),
        admission_policy="same_lineage_directional_coverage/1",
        relay_fan_in="original_chain_then_generation",
    )
